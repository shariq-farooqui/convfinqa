import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from convfinqa.dataset import load_dataset
from convfinqa.dataset.explore import analyse_dataset
from convfinqa.evaluation import DatasetEvaluator
from convfinqa.llm import OpenAIClient

app = typer.Typer(help="ConvFinQA Dataset Evaluation Tool")


@app.command()
def evaluate(
    data_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to the dataset JSON file",
    ),
    output_path: Path = typer.Option(
        "evaluation_results.json", help="Path to save evaluation results"
    ),
    filter_dataset: bool = typer.Option(
        True, help="Filter out invalid samples before evaluation"
    ),
    sample_limit: int | None = typer.Option(
        None,
        help="Limit evaluation to N samples (for testing, None=all samples)",
    ),
    batch_size: int = typer.Option(
        50, help="Number of samples to process in each batch"
    ),
    max_concurrency: int = typer.Option(
        5, help="Maximum number of concurrent API requests"
    ),
    force_restart: bool = typer.Option(
        False, help="Force restart from beginning, ignoring any saved state"
    ),
    overwrite: bool = typer.Option(
        True, help="Whether to run API calls again or just recalculate metrics"
    ),
):
    """Evaluate LLM performance on the ConvFinQA dataset."""
    if not os.getenv("OPENAI_API_KEY"):
        typer.echo("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    typer.echo(f"Loading dataset from {data_path}")
    dataset = load_dataset(file_path=data_path, filter_dataset=filter_dataset)

    sample_ids = [sample.id for sample in dataset.samples]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [id for id in sample_ids if sample_ids.count(id) > 1]
        raise ValueError(f"Dataset contains duplicate IDs: {set(duplicates)}")

    if sample_limit:
        typer.echo(
            f"Limiting evaluation to {sample_limit} samples (for testing)"
        )
        dataset.samples = dataset.samples[:sample_limit]

    llm_client = OpenAIClient()
    evaluator = DatasetEvaluator(llm_client=llm_client, dataset=dataset)

    typer.echo("Starting evaluation")
    results_file_path = f"./results/{output_path}"
    evaluator.run(
        output_path=results_file_path,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        force_restart=force_restart,
        overwrite=overwrite,
    )
    typer.echo(f"Evaluation complete. Results saved to {output_path}")


@app.command()
def explore(
    data_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to the dataset JSON file",
    ),
    show_problems: bool = typer.Option(
        False, help="Show details of problematic samples"
    ),
):
    """Analyze the dataset to understand its structure."""
    console = Console()

    typer.echo(f"Analyzing dataset from {data_path}")
    results = analyse_dataset(data_path)

    # Create summary table
    summary_table = Table(title="Dataset Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green")

    summary_table.add_row("Total Samples", str(results["total_samples"]))
    summary_table.add_row(
        "Missing Questions", str(results["missing_questions"])
    )
    summary_table.add_row("Missing Answers", str(results["missing_answers"]))

    console.print(summary_table)

    # Create answer types distribution table
    distribution_table = Table(title="Answer Types Distribution")
    distribution_table.add_column("Answer Type", style="cyan")
    distribution_table.add_column("Count", style="green")
    distribution_table.add_column("Percentage", style="yellow")

    for answer_type, count in results["answer_types"].items():
        percentage = count / results["total_samples"] * 100
        distribution_table.add_row(
            answer_type, str(count), f"{percentage:.2f}%"
        )

    console.print(distribution_table)

    if results["problematic_samples"] and show_problems:
        problem_table = Table(title="Problematic Samples")
        problem_table.add_column("ID", style="cyan")
        problem_table.add_column("Question", style="green")
        problem_table.add_column("Answer", style="yellow")

        for sample in results["problematic_samples"][:10]:  # Limit to first 10
            problem_table.add_row(
                sample["id"], sample["question"], sample["answer"]
            )

        num_problematic_samples = len(results["problematic_samples"])
        if num_problematic_samples > 10:
            console.print(
                f"Showing 10 of {num_problematic_samples} problematic samples"
            )

        console.print(problem_table)
    elif results["problematic_samples"]:
        console.print(
            f"\n[yellow]Found {num_problematic_samples} problematic samples.[/yellow]"
        )
        console.print("[yellow]Use --show-problems to see details[/yellow]")


if __name__ == "__main__":
    app()
