"""Functions for loading and processing the ConvFinQA dataset."""

import json
from pathlib import Path

from convfinqa import logger
from convfinqa.dataset.models import Dataset, QASample


def load_dataset(file_path: str | Path, filter_dataset: bool) -> Dataset:
    """Load the ConvFinQA dataset from a JSON file and convert to Pydantic model.

    Args:
        file_path: Path to the JSON dataset file.
        filter_dataset: Filters out invalid samples before returning dataset.

    Returns:
        Dataset object containing QA samples.
    """
    file_path = Path(file_path)
    logger.debug("Loading dataset", extra={"file_path": str(file_path)})

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        logger.error("File not found.", extra={"file_path": str(file_path)})
        raise e
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in file.",
            extra={"file_path": str(file_path), "error": str(e)},
        )
        raise e

    # Extract QA samples from the raw data
    samples = []
    for item in data:
        example_id = item.get("id", "unknown")

        # Extract common fields
        common_fields = {
            "pre_text": "\n".join(item.get("pre_text", [])),
            "table": item.get("table", []),
            "post_text": "\n".join(item.get("post_text", [])),
        }

        # Handle Type I (single QA pair)
        if "qa" in item:
            qa_data = item["qa"]
            samples.append(
                QASample(
                    id=example_id,
                    question=qa_data.get("question", ""),
                    answer=qa_data.get("answer"),
                    exe_ans=qa_data.get("exe_ans"),
                    **common_fields,
                )
            )

        # Handle Type II (two QA pairs)
        elif "qa_0" in item and "qa_1" in item:
            qa_0_data = item["qa_0"]
            samples.append(
                QASample(
                    id=f"{example_id}_0",
                    question=qa_0_data.get("question", ""),
                    answer=qa_0_data.get("answer"),
                    exe_ans=qa_0_data.get("exe_ans"),
                    **common_fields,
                )
            )

            # Add second QA pair
            qa_1_data = item["qa_1"]
            samples.append(
                QASample(
                    id=f"{example_id}_1",
                    question=qa_1_data.get("question", ""),
                    answer=qa_1_data.get("answer"),
                    exe_ans=qa_1_data.get("exe_ans"),
                    **common_fields,
                )
            )

    dataset = Dataset(samples=samples)
    logger.debug(
        "Loaded dataset.", extra={"total_samples": len(dataset.samples)}
    )

    if filter_dataset:
        return process_dataset(dataset)
    return dataset


def process_dataset(
    dataset: Dataset, filter_criteria: dict[str, bool] | None = None
) -> Dataset:
    """Process the dataset to remove problematic entries.

    Args:
        dataset: The dataset to process.
        filter_criteria: Dictionary specifying which types to filter out.

    Returns:
        Processed dataset.
    """
    if filter_criteria is None:
        filter_criteria = {
            "other": True,  # Remove 'other' answer types
            "boolean": True,  # Remove boolean answers
            "text_with_percentage": True,  # Remove percentages mixed with text
            "currency": True,  # Remove currency values
            "missing": True,  # Remove missing questions/answers
        }

    original_count = len(dataset.samples)

    # Filter out samples based on criteria
    filtered_samples = []

    for sample in dataset.samples:
        # Skip samples that should be filtered
        if _should_filter_sample(sample, filter_criteria):
            continue

        filtered_samples.append(sample)

    # Create new dataset with filtered samples
    processed_dataset = Dataset(samples=filtered_samples)

    # Log results
    removed_count = original_count - len(processed_dataset.samples)
    logger.debug(
        "Dataset processing complete",
        extra={
            "original_count": original_count,
            "processed_count": len(processed_dataset.samples),
            "removed_count": removed_count,
        },
    )

    return processed_dataset


def _should_filter_sample(
    sample: QASample, filter_criteria: dict[str, bool]
) -> bool:
    """Determine if a sample should be filtered based on criteria.

    Args:
        sample: The QA sample to check
        filter_criteria: Dictionary of filtering criteria

    Returns:
        True if the sample should be filtered out, False if it should be kept
    """
    # Check for missing question or answer
    if filter_criteria.get("missing", True) and not sample.is_valid:
        return True

    # Check if the answer type is in the filter criteria
    answer_type = sample.answer_type
    return filter_criteria.get(answer_type, False)
