# Display prediction results with accuracy flags
get-predictions:
  @jq -r '.samples[] | select(.predicted_answer != null) | "ID: \(.id)\nQuestion: \(.question)\nAnswer: \(.answer)\nExe_Ans: \(.exe_ans)\nPredicted: \(.predicted_answer)\nAccurate: \(.accurate)\n"' ./results/evaluation_results.json

# Show only inaccurate predictions for debugging
get-inaccurate-results:
  @jq -r '.samples[] | select(.predicted_answer != null and .accurate == false) | "ID: \(.id)\nQuestion: \(.question)\nAnswer Type: \(.answer_type)\nAnswer: \(.answer)\nExe_Ans: \(.exe_ans)\nPredicted: \(.predicted_answer)\nPred as Decimal: \(.predicted_answer / 100)\n"' ./results/evaluation_results.json

# Clean up all temporary and generated files
cleanup:
  @echo "Cleaning up temporary and generated files..."
  @find . -type d -name "__pycache__" -exec rm -rf {} +  2>/dev/null || true
  @find . -name "*.log" -delete
  @find . -name "*.pyc" -delete
  @echo "Cleanup complete!"

# Tail logs using jq
check-logs:
  @if [ -f "./convfinqa.log" ]; then \
    tail -20 ./convfinqa.log | jq -r '. | "\(.logged_at) [\(.level)] \(.message)"'; \
  elif [ -f "./logs/convfinqa.log" ]; then \
    tail -20 ./logs/convfinqa.log | jq -r '. | "\(.logged_at) [\(.level)] \(.message)"'; \
  else \
    echo "Log file not found in either ./ or ./logs/ directory"; \
  fi

# Build the application image
build:
    @echo "Building Docker image with user permissions"
    @docker build -t convfinqa:latest .

# Run container
run:
    @echo "Starting container with current directory mounted at /app"
    @docker run --rm -it \
    -e OPENAI_API_KEY \
    --mount type=bind,source="$(pwd)/convfinqa",target=/app/convfinqa \
    --volume "convfinqa_logs:/app/logs" \
    --volume "convfinqa_cache:/app/.cache" \
    --volume "convfinqa_results:/app/results" \
    convfinqa:latest

# Extract results to the local filesystem
extract-results:
    @echo "Extracting results from Docker volume to host"
    @mkdir -p ./results
    @docker run --rm -v convfinqa_results:/results -v "$(pwd)/results:/output" alpine sh -c "cp -r /results/* /output/"
