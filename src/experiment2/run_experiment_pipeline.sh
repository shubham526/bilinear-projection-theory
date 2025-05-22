#!/bin/bash
# run_experiment_pipeline.sh
# Script to run the full experiment pipeline across multiple models and datasets

# Exit on error
set -e

# Configuration
BASE_DIR="/home/username/bilinear-proj-theory"
CODE_DIR="$BASE_DIR/src/experiment2"
EMBEDDING_DIR="/home/user/sisap2025/embeddings/"
CV_TRIPLES_DIR="/home/user/sisap2025/cv_triples/"
MODEL_SAVE_DIR="/home/user/sisap2025/results/experiment2/"
CAR_DATA_DIR="/home/user/data/car"
ROBUST_DATA_DIR="/home/user/data/robust"
LOG_DIR="$BASE_DIR/logs"
DATE_STAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_run_$DATE_STAMP.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check that required directories exist
check_directory() {
  if [ ! -d "$1" ]; then
    echo "Error: Directory $1 does not exist!"
    exit 1
  fi
}

echo "Validating directories..."
check_directory "$BASE_DIR"
check_directory "$CODE_DIR"

# Create output directories if they don't exist
mkdir -p "$EMBEDDING_DIR"
mkdir -p "$CV_TRIPLES_DIR"
mkdir -p "$MODEL_SAVE_DIR"

# Verify data directories if we're using them
if [ ! -d "$CAR_DATA_DIR" ]; then
  echo "Warning: CAR_DATA_DIR at $CAR_DATA_DIR does not exist! Will use defaults."
  CAR_DATA_DIR=""
fi

if [ ! -d "$ROBUST_DATA_DIR" ]; then
  echo "Warning: ROBUST_DATA_DIR at $ROBUST_DATA_DIR does not exist! Will use defaults."
  ROBUST_DATA_DIR=""
fi

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
  echo "Warning: nvidia-smi not found. Are you sure CUDA is available?"
  echo "Continuing anyway, but you might encounter errors if CUDA is required."
fi

# Check for Python and required packages
if ! command -v python3 &> /dev/null; then
  echo "Error: Python 3 is not installed!"
  exit 1
fi

# Log start time and configuration
echo "===================================================" | tee -a "$LOG_FILE"
echo "Starting experiment pipeline at $(date)" | tee -a "$LOG_FILE"
echo "===================================================" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  BASE_DIR: $BASE_DIR" | tee -a "$LOG_FILE"
echo "  CODE_DIR: $CODE_DIR" | tee -a "$LOG_FILE"
echo "  EMBEDDING_DIR: $EMBEDDING_DIR" | tee -a "$LOG_FILE"
echo "  CV_TRIPLES_DIR: $CV_TRIPLES_DIR" | tee -a "$LOG_FILE"
echo "  MODEL_SAVE_DIR: $MODEL_SAVE_DIR" | tee -a "$LOG_FILE"
echo "  CAR_DATA_DIR: $CAR_DATA_DIR" | tee -a "$LOG_FILE"
echo "  ROBUST_DATA_DIR: $ROBUST_DATA_DIR" | tee -a "$LOG_FILE"
echo "  LOG_FILE: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===================================================" | tee -a "$LOG_FILE"

# Build the command with proper escaping for empty variables
CMD="python3 $BASE_DIR/run_experiment_pipeline.py \
  --base-dir \"$BASE_DIR\" \
  --code-dir \"$CODE_DIR\" \
  --embedding-dir \"$EMBEDDING_DIR\" \
  --cv-triples-dir \"$CV_TRIPLES_DIR\" \
  --model-save-dir \"$MODEL_SAVE_DIR\" \
  --embedding-models microsoft/mpnet-base google/electra-base facebook/contriever \
  --datasets msmarco car robust \
  --pipeline-components embeddings triples training \
  --lrb-ranks 32 64 128 \
  --include-full-rank \
  --use-chunking \
  --chunk-size 512 \
  --chunk-stride 256 \
  --chunk-aggregation hybrid \
  --device cuda \
  --continue-on-failure"

# Add data directory args only if they exist
if [ -n "$CAR_DATA_DIR" ]; then
  CMD="$CMD --car-data-dir \"$CAR_DATA_DIR\""
fi

if [ -n "$ROBUST_DATA_DIR" ]; then
  CMD="$CMD --robust-data-dir \"$ROBUST_DATA_DIR\""
fi

# Log the final command
echo "Executing command:" | tee -a "$LOG_FILE"
echo "$CMD" | tee -a "$LOG_FILE"
echo "===================================================" | tee -a "$LOG_FILE"

# Execute the command with full logging
echo "Starting execution at $(date)" | tee -a "$LOG_FILE"
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

# Check exit status
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "===================================================" | tee -a "$LOG_FILE"
  echo "Pipeline completed successfully at $(date)" | tee -a "$LOG_FILE"
else
  echo "===================================================" | tee -a "$LOG_FILE"
  echo "Pipeline execution failed with status $STATUS at $(date)" | tee -a "$LOG_FILE"
fi

# Calculate total runtime
START_TIME=$(head -n 3 "$LOG_FILE" | grep "Starting" | sed 's/.*at \(.*\)/\1/')
END_TIME=$(date)
echo "Start time: $START_TIME" | tee -a "$LOG_FILE"
echo "End time: $END_TIME" | tee -a "$LOG_FILE"
echo "===================================================" | tee -a "$LOG_FILE"

exit $STATUS