"""State management for resumable batch processing."""

import json
import os
from datetime import datetime
from pathlib import Path

from convfinqa import logger
from convfinqa.dataset.models import QASample


class StateManager:
    """Manages state for resumable processing of dataset evaluation."""

    def __init__(self, output_path: str | Path, force_restart: bool = False):
        """Initialise the state manager.

        Args:
            output_path: Path to store the state file
            force_restart: Whether to ignore existing state and start fresh
        """
        self.output_path = Path(output_path)
        self.state_path = self.output_path.with_suffix(".state.json")
        self.processed_ids: set[str] = set()
        self.current_batch = 0
        self.partial_results_path = self.state_path.with_suffix(".partial.json")

        # Try to load existing state if not forcing restart
        if self.state_path.exists() and not force_restart:
            self._load_state()
            logger.debug(
                "Resuming from previous run",
                extra={
                    "processed_samples": len(self.processed_ids),
                    "current_batch": self.current_batch,
                    "state_path": str(self.state_path),
                },
            )
        else:
            if force_restart and self.state_path.exists():
                logger.debug(
                    "Force restart requested, ignoring existing state file",
                    extra={"state_path": str(self.state_path)},
                )
            else:
                logger.debug(
                    "No existing state found, starting new evaluation run"
                )

    def _load_state(self) -> None:
        """Load existing state from file."""
        try:
            with open(self.state_path) as f:
                state = json.load(f)
                self.processed_ids = set(state.get("processed_sample_ids", []))
                self.current_batch = state.get("current_batch", 0)
                self.partial_results_path = Path(
                    state.get(
                        "partial_results_path", str(self.partial_results_path)
                    )
                )

            logger.debug(
                "State loaded successfully",
                extra={
                    "processed_ids_count": len(self.processed_ids),
                    "current_batch": self.current_batch,
                },
            )
        except Exception as e:
            logger.warning(
                "Failed to load state file, starting fresh",
                extra={"error": str(e), "state_path": str(self.state_path)},
            )
            self.processed_ids = set()
            self.current_batch = 0

    def get_unprocessed_samples(
        self, all_samples: list[QASample]
    ) -> list[QASample]:
        """Get samples that haven't been processed yet.

        Args:
            all_samples: Complete list of samples

        Returns:
            List of unprocessed samples
        """
        unprocessed = [s for s in all_samples if s.id not in self.processed_ids]
        logger.debug(
            "Filtered unprocessed samples",
            extra={
                "total_samples": len(all_samples),
                "processed_samples": len(self.processed_ids),
                "unprocessed_samples": len(unprocessed),
            },
        )
        return unprocessed

    def mark_processed(self, sample_id: str) -> None:
        """Mark a sample as processed.

        Args:
            sample_id: ID of the processed sample
        """
        self.processed_ids.add(sample_id)

    def save_checkpoint(self) -> None:
        """Save the current state to file."""
        state = {
            "processed_sample_ids": list(self.processed_ids),
            "current_batch": self.current_batch,
            "last_batch_timestamp": datetime.now().isoformat(),
            "partial_results_path": str(self.partial_results_path),
        }

        # Create directory if it doesn't exist
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first to avoid corruption if interrupted
        temp_path = f"{self.state_path}.tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(state, f)

            # Replace the old state file with the new one
            os.replace(temp_path, self.state_path)

            self.current_batch += 1
            logger.debug(
                "Saved checkpoint",
                extra={
                    "batch": self.current_batch - 1,
                    "processed_samples": len(self.processed_ids),
                    "state_path": str(self.state_path),
                },
            )
        except Exception as e:
            logger.error(
                "Failed to save checkpoint",
                extra={"error": str(e), "state_path": str(self.state_path)},
            )

    def cleanup(self) -> None:
        """Clean up state files after successful completion."""
        try:
            if self.state_path.exists():
                self.state_path.unlink()
                logger.debug(
                    "Removed state file", extra={"path": str(self.state_path)}
                )

            if self.partial_results_path.exists():
                self.partial_results_path.unlink()
                logger.debug(
                    "Removed partial results file",
                    extra={"path": str(self.partial_results_path)},
                )

            logger.debug("Cleaned up state files")
        except Exception as e:
            logger.warning(
                "Failed to clean up state files", extra={"error": str(e)}
            )
