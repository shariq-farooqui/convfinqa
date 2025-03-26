# ConvFinQA Evaluator

A system for evaluating LLM performance on numerical reasoning tasks using the ConvFinQA dataset.

## Overview

This project implements an evaluation framework for assessing Language Model performance on financial numerical reasoning tasks. It uses the [ConvFinQA dataset](https://github.com/czyssrs/ConvFinQA) to benchmark how well LLMs can interpret financial documents containing text and tables, extract relevant information, and perform calculations to answer numerical questions.

Key aspects of the system:
- Asynchronous processing for LLM API requests
- Evaluation metrics (accuracy, MAE, RMSE, etc.)
- Resumable processing with checkpointing
- Docker support for reproducible evaluation

## Installation

### Prerequisites
- Docker
- [Just](https://github.com/casey/just) command runner (optional, for convenience)
- [Direnv](https://direnv.net/docs/installation.html) (optional, for environment variable management)
- [jq](https://jqlang.github.io/jq/download/) (optional, for inspecting JSON results)

Note: If you use Docker, both `just` and `jq` will be automatically installed in the container.

Before running, you'll need to set up your OpenAI API key. You can either:
- Copy `.envrc.example` to `.envrc`, add your OpenAI API key, and use direnv to load it
- Or simply export the key in your current shell: `export OPENAI_API_KEY=your-key-here`

### Option 1: Using Docker (Recommended)

Build the Docker image:
```bash
just build
# Or directly with Docker
docker build -t convfinqa:latest .
```

Run the container:
```bash
just run
# Or directly with Docker
docker run --rm -it \
  -e OPENAI_API_KEY \
  --mount type=bind,source="$(pwd)/convfinqa",target=/app/convfinqa \
  --volume "convfinqa_logs:/app/logs" \
  --volume "convfinqa_cache:/app/.cache" \
  --volume "convfinqa_results:/app/results" \
  convfinqa:latest
```

### Option 2: Local Installation

1. Install Python 3.10 or higher
2. Clone this repository
3. Install dependencies:
```bash
pip install .
```

## Usage

When you run the container, it will display a help message showing available commands and options. Here are some example commands to get started:

### Docker Container

```bash
# Run the evaluation on the full dataset
python -m convfinqa evaluate data/train.json

# Run on a limited number of samples (for testing)
python -m convfinqa evaluate --sample-limit 10 data/train.json

# Explore the dataset structure
python -m convfinqa explore data/train.json --show-problems
```

## System Architecture

The system consists of several key components:

1. **Dataset Module**: Loads and processes the ConvFinQA dataset
2. **LLM Client**: Interfaces with OpenAI's API for generating responses
3. **Evaluation Module**: Processes samples and calculates performance metrics
4. **Utils**: Expression evaluation and state management
5. **Logging**: Structured logging for tracking execution

The evaluation process:
1. Load and preprocess the dataset
2. Process samples in batches with controlled concurrency
3. Save checkpoints for resumable processing
4. Calculate and display evaluation metrics

## Initial Results

Running the evaluation with GPT-4o mini produces these initial metrics:

| Metric | Value |
|--------|-------|
| Accuracy | 0.7389 |
| Mean Absolute Error (MAE) | 10,293,458,766.0988 |
| Root Mean Squared Error (RMSE) | 607,882,126,924.4243 |
| R² Score | -0.0003 |
| Mean Absolute Percentage Error (MAPE) | 1,593,161.46% |

The contrast between good accuracy but poor error metrics indicates that while most predictions are close enough to be considered correct (within the small thresholds used), a small number of extreme outliers are severely skewing the error metrics. These outliers likely result from:

- Order of magnitude errors (correct number but wrong power of 10)
- Decimal vs. percentage format confusion
- Expression parsing issues with large numbers
- Formatting inconsistencies between answers and predictions

For a more representative assessment, future work should include outlier detection and mitigation. Adjusting accuracy thresholds could also affect the reported accuracy values.

## Utility Commands

If you've run evaluations inside the container but want to work with the results outside, first extract them:
```bash
just extract-results
```

View predictions:
```bash
just get-predictions
# Shows sample IDs, questions, answers, and predictions with accuracy flags
```

View inaccurate results:
```bash
just get-inaccurate-results
# Displays only incorrect predictions for debugging purposes
```

Check logs:
```bash
just check-logs
# Shows recent log entries with timestamps and log levels
```
