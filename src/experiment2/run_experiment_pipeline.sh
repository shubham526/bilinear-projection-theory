#!/bin/bash
# run_experiment_pipeline.sh
# FIXED VERSION - Properly calls the Python pipeline script

# Exit on error
set -e

# Configuration
BASE_DIR="/home/user/bilinear-projection-theory"
CODE_DIR="$BASE_DIR/src/experiment2/"
SISAP_DIR="/home/user/sisap2025"
EMBEDDING_DIR="$SISAP_DIR/embeddings/"
CV_TRIPLES_DIR="$SISAP_DIR/cv_triples/"
MODEL_SAVE_DIR="$SISAP_DIR/results/experiment2/"
CAR_DATA_DIR="$SISAP_DIR/data/car/"
ROBUST_DATA_DIR="$SISAP_DIR/data/robust/"
LOG_DIR="$SISAP_DIR/logs"
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

# Verify data directories and warn if missing
if [ ! -d "$CAR_DATA_DIR" ]; then
  echo "Warning: CAR_DATA_DIR at $CAR_DATA_DIR does not exist! Some functionality may be limited."
fi

if [ ! -d "$ROBUST_DATA_DIR" ]; then
  echo "Warning: ROBUST_DATA_DIR at $ROBUST_DATA_DIR does not exist! Some functionality may be limited."
fi

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
  echo "Warning: nvidia-smi not found. Will use CPU instead of CUDA."
  DEVICE="cpu"
else
  DEVICE="cuda"
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
echo "  SISAP_DIR: $SISAP_DIR" | tee -a "$LOG_FILE"
echo "  EMBEDDING_DIR: $EMBEDDING_DIR" | tee -a "$LOG_FILE"
echo "  CV_TRIPLES_DIR: $CV_TRIPLES_DIR" | tee -a "$LOG_FILE"
echo "  MODEL_SAVE_DIR: $MODEL_SAVE_DIR" | tee -a "$LOG_FILE"
echo "  CAR_DATA_DIR: $CAR_DATA_DIR" | tee -a "$LOG_FILE"
echo "  ROBUST_DATA_DIR: $ROBUST_DATA_DIR" | tee -a "$LOG_FILE"
echo "  DEVICE: $DEVICE" | tee -a "$LOG_FILE"
echo "  LOG_FILE: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===================================================" | tee -a "$LOG_FILE"

# Build the command for the Python pipeline script
CMD="python3 $CODE_DIR/run_experiment_pipeline.py \
  --base-dir \"$SISAP_DIR\" \
  --code-dir \"$CODE_DIR\" \
  --embedding-dir \"$EMBEDDING_DIR\" \
  --cv-triples-dir \"$CV_TRIPLES_DIR\" \
  --model-save-dir \"$MODEL_SAVE_DIR\" \
  --embedding-models facebook/contriever \
  --datasets car robust \
  --pipeline-components embeddings triples training \
  --lrb-ranks 32 64 128 \
  --include-full-rank \
  --device $DEVICE \
  --continue-on-failure \
  --verbose"

# Add data directory args if they exist
if [ -d "$CAR_DATA_DIR" ]; then
  CMD="$CMD --car-data-dir \"$CAR_DATA_DIR\""
fi

if [ -d "$ROBUST_DATA_DIR" ]; then
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
  echo "Start time: Thu May 22 20:03:18 UTC 2025" | tee -a "$LOG_FILE"
  echo "End time: $(date)" | tee -a "$LOG_FILE"
else
  echo "===================================================" | tee -a "$LOG_FILE"
  echo "Pipeline execution failed with status $STATUS at $(date)" | tee -a "$LOG_FILE"
  echo "Check the logs above for details" | tee -a "$LOG_FILE"
fi

echo "===================================================" | tee -a "$LOG_FILE"

exit $STATUS