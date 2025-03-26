import sys
from pathlib import Path
from typing import Any

from convfinqa import logger
from convfinqa.dataset.loader import load_dataset


def analyse_dataset(file_path: str | Path) -> dict[str, Any]:
    """Analyse the ConvFinQA dataset to understand answer formats and identify issues.

    Args:
        file_path: Path to the dataset JSON file

    Returns:
        Dictionary with analysis results
    """
    dataset = load_dataset(file_path=file_path, filter_dataset=False)

    logger.debug(
        "Starting dataset analysis.",
        extra={"total_samples": len(dataset.samples)},
    )

    results = {
        "total_samples": len(dataset.samples),
        "missing_questions": 0,
        "missing_answers": 0,
        "answer_types": {
            "numeric": 0,
            "clean_percentage": 0,
            "text_with_percentage": 0,
            "currency": 0,
            "boolean": 0,
            "other": 0,
            "invalid": 0,
        },
        "problematic_samples": [],
    }

    for sample in dataset.samples:
        # Check validity
        if not sample.question:
            results["missing_questions"] += 1

        if sample.answer is None or sample.answer == "":
            results["missing_answers"] += 1

        # Count answer types
        answer_type = sample.answer_type
        results["answer_types"][answer_type] += 1

        # Record problematic samples
        if answer_type == "other" and sample.is_valid:
            results["problematic_samples"].append(
                {
                    "id": sample.id,
                    "question": sample.question,
                    "answer": sample.answer,
                }
            )

    # Log summary
    logger.debug(
        "Analysis complete.",
        extra={
            "total_samples": results["total_samples"],
            "numeric_answers": results["answer_types"]["numeric"],
            "clean_percentage_answers": results["answer_types"][
                "clean_percentage"
            ],
            "text_with_percentage_answers": results["answer_types"][
                "text_with_percentage"
            ],
            "currency_answers": results["answer_types"]["currency"],
            "boolean_answers": results["answer_types"]["boolean"],
            "other_answers": results["answer_types"]["other"],
            "invalid_answers": results["answer_types"]["invalid"],
        },
    )

    if results["problematic_samples"]:
        logger.warning(
            "Found problematic non-numeric answers.",
            extra={"count": len(results["problematic_samples"])},
        )

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore.py <data_file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    results = analyse_dataset(sys.argv[1])
    problematic_answers = results.get("problematic_samples")
    if len(problematic_answers) > 0:
        for problem in problematic_answers:
            print(problem)
