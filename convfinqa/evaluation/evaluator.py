import asyncio
import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from convfinqa import logger
from convfinqa.dataset.models import Dataset, QASample
from convfinqa.llm import LLMClient
from convfinqa.llm.prompts import SYSTEM_PROMPT
from convfinqa.utils import ExpressionEvaluator, StateManager


class DatasetEvaluator:
    """Orchestrate evaluation of the ConvFinQA dataset."""

    def __init__(self, llm_client: LLMClient, dataset: Dataset):
        self._llm_client = llm_client
        self._dataset = dataset
        self._parser = ExpressionEvaluator()
        self._console = Console()

    async def _process_sample(self, sample: QASample) -> bool:
        """Process a single sample through the LLM."""
        try:
            response = await self._llm_client.generate_response(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=sample.formatted_context,
            )
            sample.llm_response = response.choices[0].message.parsed

            # Calculate the predicted answer using the right precision
            sample.predicted_answer = self._parser.evaluate(
                sample.llm_response.expression, sample.decimal_precision
            )

            sample.accurate = self._is_answer_accurate(sample)
            return True
        except Exception as e:
            logger.error(
                "Failed to process sample",
                extra={"sample_id": sample.id, "error": str(e)},
            )
            return False

    async def _process_batch(
        self, batch: list[QASample], max_concurrency: int
    ) -> int:
        """Process a batch of samples with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_with_semaphore(sample):
            async with semaphore:
                return await self._process_sample(sample)

        tasks = [process_with_semaphore(sample) for sample in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r is True)
        return successful

    async def _process_in_batches(
        self, state_manager: StateManager, batch_size: int, max_concurrency: int
    ):
        """Process dataset samples in batches with checkpoint functionality."""
        unprocessed_samples = state_manager.get_unprocessed_samples(
            self._dataset.samples
        )
        total_unprocessed = len(unprocessed_samples)

        if total_unprocessed == 0:
            logger.debug("No unprocessed samples to process")
            return

        total_batches = (total_unprocessed + batch_size - 1) // batch_size
        logger.debug(
            "Starting batch processing",
            extra={
                "total_unprocessed": total_unprocessed,
                "batch_size": batch_size,
                "max_concurrency": max_concurrency,
                "total_batches": total_batches,
            },
        )

        with tqdm(total=total_unprocessed, desc="Processing samples") as pbar:
            for i in range(0, total_unprocessed, batch_size):
                current_batch_num = i // batch_size + 1
                logger.debug(
                    f"Processing batch {current_batch_num}/{total_batches}",
                    extra={"batch_start_index": i},
                )

                batch = unprocessed_samples[i : i + batch_size]

                # Process this batch with concurrent requests
                successful_samples = await self._process_batch(
                    batch, max_concurrency
                )

                # Mark processed samples
                for sample in batch:
                    if sample.llm_response is not None:
                        state_manager.mark_processed(sample.id)

                # Update progress bar
                pbar.update(len(batch))

                # Save state after each batch
                state_manager.save_checkpoint()

                # Save partial results
                self._save_results(state_manager.partial_results_path)

                logger.debug(
                    f"Completed batch {current_batch_num}/{total_batches}",
                    extra={
                        "successful_samples": successful_samples,
                        "batch_size": len(batch),
                    },
                )

        logger.debug(
            "Batch processing complete",
            extra={"total_processed": len(state_manager.processed_ids)},
        )

    def run(
        self,
        output_path: str | Path,
        batch_size: int = 50,
        max_concurrency: int = 5,
        force_restart: bool = False,
        overwrite: bool = True,
    ) -> None:
        """Run evaluation on the ConvFinQA Dataset.

        Args:
            output_path: File path to store the evaluation results
            batch_size: Number of samples to process before saving state
            max_concurrency: Maximum number of concurrent API requests
            force_restart: Whether to ignore existing state and start fresh
            overwrite: Whether to run the API calls again or use existing results
        """
        output_path = Path(output_path)
        # Check if we are recalculating metrics using results from a previous run
        if not overwrite and output_path.exists():
            logger.debug("Using existing results file for metrics calculation")
            self._load_results_and_calculate_metrics(output_path)
            return

        # Setup checkpoint state
        state_manager = StateManager(output_path, force_restart)

        # Process samples in batches
        asyncio.run(
            self._process_in_batches(state_manager, batch_size, max_concurrency)
        )

        # Save final results
        self._save_results(output_path)

        # Calculate and display metrics
        metrics = self._calculate_metrics()
        self._display_metrics(metrics)

        # Cleanup state file after successful completion
        state_manager.cleanup()

    def _save_results(self, output_path: str) -> None:
        """Save the evaluation results to a JSON file."""
        # Create a serialisable version of the dataset
        serialised_data = self._dataset.model_dump_json(indent=4)
        with open(output_path, "w") as f:
            f.write(serialised_data)

        logger.debug("Results saved", extra={"output_path": str(output_path)})

    def _calculate_metrics(self) -> dict[str, float | str]:
        """Calculate evaluation metrics for the dataset.

        Returns:
            Dictionary containing various performance metrics.
        """
        # Get samples with predictions
        samples_with_predictions = [
            s for s in self._dataset.samples if s.predicted_answer is not None
        ]

        # Count samples with LLM responses but no predictions (parse errors)
        samples_with_response_only = [
            s
            for s in self._dataset.samples
            if s.llm_response is not None and s.predicted_answer is None
        ]

        total_samples = len(self._dataset.samples)
        processed_count = len(samples_with_predictions)

        # Log parsing errors if any exist
        if samples_with_response_only:
            logger.warning(
                "Some samples have LLM responses but no predictions",
                extra={"count": len(samples_with_response_only)},
            )

        # Create metrics dictionary with basic counts
        metrics = {
            "total_samples": total_samples,
            "processed_samples": processed_count,
            "processing_rate": processed_count / total_samples
            if total_samples > 0
            else 0,
            "accuracy": self._calculate_accuracy(samples_with_predictions),
        }

        # Only calculate numeric metrics if we have compatible samples
        numeric_samples = [
            s
            for s in samples_with_predictions
            if s.answer_type in ["clean_percentage", "numeric"]
        ]

        # Add debug information about sample types
        answer_type_counts = {}
        for s in samples_with_predictions:
            answer_type_counts[s.answer_type] = (
                answer_type_counts.get(s.answer_type, 0) + 1
            )

        logger.debug(
            "Sample distribution by answer type",
            extra={
                "answer_type_counts": answer_type_counts,
                "numeric_samples_count": len(numeric_samples),
            },
        )

        if numeric_samples:
            try:
                true_values, pred_values = self._get_numeric_pairs(
                    numeric_samples
                )

                logger.debug(
                    "Numeric pairs extracted",
                    extra={
                        "true_values_count": len(true_values),
                        "pred_values_count": len(pred_values),
                        "true_values_sample": str(true_values[:5])
                        if true_values
                        else "[]",
                        "pred_values_sample": str(pred_values[:5])
                        if pred_values
                        else "[]",
                    },
                )

                # Calculate error metrics with error handling
                try:
                    metrics["mae"] = self._calculate_mae(
                        true_values, pred_values
                    )
                except Exception as e:
                    logger.error(
                        "Error calculating MAE", extra={"error": str(e)}
                    )
                    metrics["mae"] = "N/A"

                try:
                    metrics["rmse"] = self._calculate_rmse(
                        true_values, pred_values
                    )
                except Exception as e:
                    logger.error(
                        "Error calculating RMSE", extra={"error": str(e)}
                    )
                    metrics["rmse"] = "N/A"

                try:
                    metrics["r2"] = self._calculate_r2(true_values, pred_values)
                except Exception as e:
                    logger.error(
                        "Error calculating R²", extra={"error": str(e)}
                    )
                    metrics["r2"] = "N/A"

                try:
                    metrics["mape"] = self._calculate_mape(
                        true_values, pred_values
                    )
                except Exception as e:
                    logger.error(
                        "Error calculating MAPE", extra={"error": str(e)}
                    )
                    metrics["mape"] = "N/A"
            except Exception as e:
                logger.error(
                    "Error processing numeric samples",
                    extra={"error": str(e)},
                )
                metrics.update(
                    {
                        "mae": "N/A",
                        "rmse": "N/A",
                        "r2": "N/A",
                        "mape": "N/A",
                    }
                )
        else:
            logger.warning(
                "No numeric samples available for error metrics calculation",
                extra={"processed_samples": processed_count},
            )
            metrics.update(
                {
                    "mae": "N/A",
                    "rmse": "N/A",
                    "r2": "N/A",
                    "mape": "N/A",
                }
            )

        return metrics

    def _calculate_accuracy(self, samples: list[QASample]) -> float:
        """Calculate the accuracy of predictions.

        Args:
            samples: List of samples with predictions

        Returns:
            Accuracy as a float between 0 and 1
        """
        if not samples:
            return 0.0

        accurate_count = sum(1 for s in samples if s.accurate is True)
        return round(accurate_count / len(samples), 4)

    def _get_numeric_pairs(
        self, samples: list[QASample]
    ) -> tuple[list[float], list[float]]:
        """Extract pairs of true and predicted values for numeric metrics.

        Args:
            samples: List of samples with numeric or percentage answers

        Returns:
            Tuple of (true_values, predicted_values) lists
        """
        true_values = []
        pred_values = []

        for sample in samples:
            # For percentage answers with exe_ans
            if (
                sample.answer_type == "clean_percentage"
                and sample.exe_ans is not None
            ):
                true_value = sample.exe_ans
                pred_value = sample.predicted_answer / 100  # Convert to decimal
            # For percentage answers without exe_ans
            elif sample.answer_type == "clean_percentage":
                true_value = (
                    float(sample.answer.strip("%").replace(",", "")) / 100
                )
                pred_value = sample.predicted_answer / 100  # Both as decimal
            # For numeric answers
            else:  # numeric
                true_value = float(sample.answer.replace(",", ""))
                pred_value = sample.predicted_answer

            true_values.append(true_value)
            pred_values.append(pred_value)

        return true_values, pred_values

    def _calculate_mae(
        self, true_values: list[float], pred_values: list[float]
    ) -> float:
        """Calculate Mean Absolute Error.

        Args:
            true_values: List of ground truth values
            pred_values: List of predicted values

        Returns:
            MAE as a float
        """
        if not true_values:
            return 0.0

        return float(
            np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
        )

    def _calculate_rmse(
        self, true_values: list[float], pred_values: list[float]
    ) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            true_values: List of ground truth values
            pred_values: List of predicted values

        Returns:
            RMSE as a float
        """
        if not true_values:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(np.array(true_values) - np.array(pred_values))
                )
            )
        )

    def _calculate_r2(
        self, true_values: list[float], pred_values: list[float]
    ) -> float | str:
        """Calculate R-squared (coefficient of determination).

        Args:
            true_values: List of ground truth values
            pred_values: List of predicted values

        Returns:
            R-squared as a float, or "N/A" if calculation is invalid
        """
        if not true_values:
            return "N/A"

        true_array = np.array(true_values)
        ss_total = np.sum(np.square(true_array - np.mean(true_array)))
        ss_residual = np.sum(np.square(true_array - np.array(pred_values)))

        if ss_total == 0:
            return "N/A"

        return float(1 - (ss_residual / ss_total))

    def _calculate_mape(
        self, true_values: list[float], pred_values: list[float]
    ) -> float | str:
        """Calculate Mean Absolute Percentage Error.

        Args:
            true_values: List of ground truth values
            pred_values: List of predicted values

        Returns:
            MAPE as a float, or "N/A" if calculation is invalid
        """
        if not true_values:
            return "N/A"

        true_array = np.array(true_values)
        pred_array = np.array(pred_values)

        # Avoid division by zero
        non_zero_indices = true_array != 0
        if not np.any(non_zero_indices):
            return "N/A"

        mape = (
            np.mean(
                np.abs(
                    (
                        true_array[non_zero_indices]
                        - pred_array[non_zero_indices]
                    )
                    / true_array[non_zero_indices]
                )
            )
            * 100
        )

        return float(mape)

    def _is_answer_accurate(self, sample: QASample) -> bool:
        """Determine if the predicted answer is accurate for this sample."""
        if sample.predicted_answer is None:
            return False

        # For percentage answers with exe_ans
        if (
            sample.answer_type == "clean_percentage"
            and sample.exe_ans is not None
        ):
            # For percentage answers, use scaled comparison with exe_ans
            prediction_as_decimal = sample.predicted_answer / 100
            exact_match = abs(prediction_as_decimal - sample.exe_ans) < 1e-4

            # Also check direct percentage comparison as fallback
            percentage_match = False
            if not exact_match:
                ground_truth = float(sample.answer.strip("%").replace(",", ""))
                percentage_match = (
                    abs(sample.predicted_answer - ground_truth) < 1e-2
                )

            return exact_match or percentage_match

        # For percentage answers without exe_ans
        elif sample.answer_type == "clean_percentage":
            ground_truth = float(sample.answer.strip("%").replace(",", ""))
            return abs(sample.predicted_answer - ground_truth) < 1e-2

        # For numeric answers
        elif sample.answer_type == "numeric":
            ground_truth = float(sample.answer.replace(",", ""))
            return abs(sample.predicted_answer - ground_truth) < 1e-3

        # Fallback for other types
        else:
            # Try numeric comparison as last resort
            try:
                return (
                    abs(float(sample.predicted_answer) - float(sample.answer))
                    < 1e-3
                )
            except ValueError:
                return str(sample.predicted_answer) == sample.answer

    def _display_metrics(self, metrics: dict[str, str | float]):
        """Display calculated metrics in a well formatted table."""
        # Overall statistics table
        stats_table = Table(title="Evaluation Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Samples", str(metrics["total_samples"]))
        stats_table.add_row(
            "Processed Samples", str(metrics["processed_samples"])
        )
        stats_table.add_row(
            "Success Rate",
            f"{metrics['processed_samples'] / metrics['total_samples'] * 100:.2f}%",
        )

        # Performance metrics table
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics_table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        metrics_table.add_row(
            "Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}"
        )
        metrics_table.add_row(
            "Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.4f}"
        )

        if metrics["r2"] != "N/A":
            metrics_table.add_row("R² Score", f"{metrics['r2']:.4f}")
        else:
            metrics_table.add_row("R² Score", "N/A")

        if metrics["mape"] != "N/A":
            metrics_table.add_row(
                "Mean Absolute Percentage Error (MAPE)",
                f"{metrics['mape']:.2f}%",
            )
        else:
            metrics_table.add_row(
                "Mean Absolute Percentage Error (MAPE)", "N/A"
            )

        self._console.print(stats_table)
        self._console.print(metrics_table)

    def _load_results_and_calculate_metrics(self, output_path: Path) -> None:
        """Load existing results and calculate metrics without running LLM calls.

        Args:
            output_path: Path to the existing results file
        """
        logger.debug(
            "Loading existing results file", extra={"path": str(output_path)}
        )

        try:
            with open(output_path) as f:
                # Parse the JSON data and reconstruct the Dataset object
                data = json.load(f)

                # Replace the current dataset with the loaded one
                self._dataset = Dataset(**data)

                # Count samples with predictions
                samples_with_predictions = [
                    s
                    for s in self._dataset.samples
                    if s.predicted_answer is not None
                ]
                logger.debug(
                    "Loaded results file",
                    extra={
                        "total_samples": len(self._dataset.samples),
                        "samples_with_predictions": len(
                            samples_with_predictions
                        ),
                    },
                )

                # Calculate and display metrics
                metrics = self._calculate_metrics()
                self._display_metrics(metrics)

        except Exception as e:
            logger.error(
                "Failed to load results file",
                extra={"path": str(output_path), "error": str(e)},
            )
            raise
