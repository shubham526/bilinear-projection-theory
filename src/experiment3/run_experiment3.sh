#!/bin/bash

# experiment3_batch_runner.sh
# Batch runner for Experiment 3: Low-Rank Approximation Analysis

set -e  # Exit on any error

# Configuration
BASE_DIR="/home/user/sisap2025"
RESULTS_BASE_DIR="$BASE_DIR/results"
EXP2_RESULTS_DIR="$RESULTS_BASE_DIR/experiment2"
EXP3_RESULTS_DIR="$RESULTS_BASE_DIR/experiment3"
SCRIPT_PATH="/home/user/bilinear-projection-theory/src/experiment3/experiment3.py"  # Adjust path as needed

# Arrays for datasets and models
datasets=("car" "robust" "msmarco")
models=("bert-base-uncased" "facebook-contriever" "microsoft-mpnet-base")

# Experiment parameters
ranks=(1 2 4 8 16 32 64 128)
device="cuda"
num_samples=100000
model_type="full_rank_bilinear"

# Logging setup
LOG_DIR="$EXP3_RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_run_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/batch_errors_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$MAIN_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$MAIN_LOG" | tee -a "$ERROR_LOG"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "$MAIN_LOG"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Python script exists
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        log_error "Python script not found: $SCRIPT_PATH"
        log_error "Please adjust SCRIPT_PATH in the script configuration"
        exit 1
    fi

    # Check if base directories exist
    if [[ ! -d "$EXP2_RESULTS_DIR" ]]; then
        log_error "Experiment 2 results directory not found: $EXP2_RESULTS_DIR"
        exit 1
    fi

    # Check CUDA availability if using GPU
    if [[ "$device" == "cuda" ]]; then
        if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log_warn "CUDA not available, falling back to CPU"
            device="cpu"
        else
            log_info "CUDA is available"
        fi
    fi

    # Create output directories
    mkdir -p "$EXP3_RESULTS_DIR"

    log_info "Prerequisites check completed"
}

# Function to check if model file exists
check_model_file() {
    local model_path="$1"
    if [[ -f "$model_path" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to run single experiment
run_experiment() {
    local dataset="$1"
    local model="$2"
    local model_path="$3"
    local output_dir="$4"

    log_info "Starting experiment: Dataset=$dataset, Model=$model"
    log_debug "Model path: $model_path"
    log_debug "Output dir: $output_dir"

    # Create output directory
    mkdir -p "$output_dir"

    # Prepare ranks argument
    local ranks_str="${ranks[*]}"

    # Run the experiment
    local cmd="python \"$SCRIPT_PATH\" \
        --model_path \"$model_path\" \
        --dataset \"$dataset\" \
        --ranks $ranks_str \
        --results_dir \"$output_dir\" \
        --device \"$device\" \
        --num_samples $num_samples \
        --verify_bounds"

    log_debug "Command: $cmd"

    # Execute with timeout and capture output
    local exp_log="$output_dir/experiment.log"
    local exp_error="$output_dir/experiment_error.log"

    if timeout 3600 bash -c "$cmd" > "$exp_log" 2> "$exp_error"; then
        log_info "✓ Experiment completed successfully: Dataset=$dataset, Model=$model"

        # Check if results were generated
        if [[ -f "$output_dir/experiment3_summary.json" ]]; then
            log_info "✓ Results summary generated: $output_dir/experiment3_summary.json"
        else
            log_warn "⚠ No results summary found (experiment may have failed silently)"
        fi

        return 0
    else
        local exit_code=$?
        log_error "✗ Experiment failed: Dataset=$dataset, Model=$model (exit code: $exit_code)"
        log_error "Check logs: $exp_log and $exp_error"

        # Show last few lines of error log if it exists
        if [[ -f "$exp_error" && -s "$exp_error" ]]; then
            log_error "Last few lines of error log:"
            tail -5 "$exp_error" | while read line; do
                log_error "  $line"
            done
        fi

        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    log_info "Generating batch run summary..."

    local summary_file="$EXP3_RESULTS_DIR/batch_summary_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "Experiment 3 Batch Run Summary"
        echo "=============================="
        echo "Date: $(date)"
        echo "Total combinations: $((${#datasets[@]} * ${#models[@]}))"
        echo ""
        echo "Configuration:"
        echo "- Datasets: ${datasets[*]}"
        echo "- Models: ${models[*]}"
        echo "- Ranks: ${ranks[*]}"
        echo "- Device: $device"
        echo "- Samples: $num_samples"
        echo ""
        echo "Results:"

        local total=0
        local successful=0
        local failed=0

        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do
                total=$((total + 1))
                local output_dir="$EXP3_RESULTS_DIR/$dataset/$model"

                if [[ -f "$output_dir/experiment3_summary.json" ]]; then
                    successful=$((successful + 1))
                    echo "✓ $dataset/$model - SUCCESS"
                else
                    failed=$((failed + 1))
                    echo "✗ $dataset/$model - FAILED"
                fi
            done
        done

        echo ""
        echo "Summary Statistics:"
        echo "- Total experiments: $total"
        echo "- Successful: $successful"
        echo "- Failed: $failed"
        echo "- Success rate: $(( successful * 100 / total ))%"

    } | tee "$summary_file"

    log_info "Summary saved to: $summary_file"
}

# Main execution function
main() {
    log_info "Starting Experiment 3 batch runner"
    log_info "Timestamp: $(date)"
    log_info "Configuration: ${#datasets[@]} datasets × ${#models[@]} models = $((${#datasets[@]} * ${#models[@]})) total experiments"

    # Check prerequisites
    check_prerequisites

    # Track statistics
    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    local skipped_experiments=0

    # Main execution loop
    for dataset in "${datasets[@]}"; do
        log_info "Processing dataset: $dataset"

        for model in "${models[@]}"; do
            total_experiments=$((total_experiments + 1))

            # Construct paths
            local model_path="$EXP2_RESULTS_DIR/$model/$dataset/$model_type/best_model.pth"
            local output_dir="$EXP3_RESULTS_DIR/$dataset/$model"

            # Check if model file exists
            if ! check_model_file "$model_path"; then
                log_warn "⚠ Model file not found, skipping: $model_path"
                skipped_experiments=$((skipped_experiments + 1))
                continue
            fi

            # Check if experiment already completed (optional skip)
            if [[ -f "$output_dir/experiment3_summary.json" ]]; then
                log_info "⚠ Experiment already completed, skipping: $dataset/$model"
                log_info "  (Delete $output_dir/experiment3_summary.json to re-run)"
                skipped_experiments=$((skipped_experiments + 1))
                continue
            fi

            # Run the experiment
            if run_experiment "$dataset" "$model" "$model_path" "$output_dir"; then
                successful_experiments=$((successful_experiments + 1))
            else
                failed_experiments=$((failed_experiments + 1))
            fi

            # Brief pause between experiments
            sleep 2
        done
    done

    # Final summary
    log_info "Batch run completed!"
    log_info "Statistics:"
    log_info "  Total experiments: $total_experiments"
    log_info "  Successful: $successful_experiments"
    log_info "  Failed: $failed_experiments"
    log_info "  Skipped: $skipped_experiments"
    log_info "  Success rate: $(( successful_experiments * 100 / (total_experiments - skipped_experiments) ))%"

    # Generate detailed summary
    generate_summary

    log_info "Logs saved to:"
    log_info "  Main log: $MAIN_LOG"
    log_info "  Error log: $ERROR_LOG"

    # Exit with appropriate code
    if [[ $failed_experiments -gt 0 ]]; then
        log_warn "Some experiments failed. Check error logs for details."
        exit 1
    else
        log_info "All experiments completed successfully!"
        exit 0
    fi
}

# Signal handlers for graceful shutdown
cleanup() {
    log_warn "Script interrupted. Cleaning up..."
    exit 130
}

trap cleanup SIGINT SIGTERM

# Help function
show_help() {
    cat << EOF
Experiment 3 Batch Runner

Usage: $0 [OPTIONS]

This script runs Experiment 3 (Low-Rank Approximation Analysis) across
multiple datasets and models in batch mode.

Options:
    -h, --help          Show this help message
    --dry-run          Show what would be executed without running
    --force            Overwrite existing results
    --datasets LIST    Comma-separated list of datasets (default: car,robust,msmarco)
    --models LIST      Comma-separated list of models (default: bert-base-uncased,facebook-contriever,microsoft-mpnet-base)
    --device DEVICE    Device to use (default: cuda)
    --samples N        Number of samples for verification (default: 100000)

Examples:
    $0                          # Run with default settings
    $0 --dry-run               # Show what would be executed
    $0 --datasets car,msmarco  # Run only on specific datasets
    $0 --device cpu            # Use CPU instead of GPU

Configuration can be modified by editing the script variables at the top.
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --datasets)
            IFS=',' read -ra datasets <<< "$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra models <<< "$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --samples)
            num_samples="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Handle dry run
if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN MODE - Commands that would be executed:"
    echo "=============================================="

    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            model_path="$EXP2_RESULTS_DIR/$model/$dataset/$model_type/best_model.pth"
            output_dir="$EXP3_RESULTS_DIR/$dataset/$model"
            ranks_str="${ranks[*]}"

            echo ""
            echo "Dataset: $dataset, Model: $model"
            echo "Command: python \"$SCRIPT_PATH\" \\"
            echo "    --model_path \"$model_path\" \\"
            echo "    --dataset \"$dataset\" \\"
            echo "    --ranks $ranks_str \\"
            echo "    --results_dir \"$output_dir\" \\"
            echo "    --device \"$device\" \\"
            echo "    --num_samples $num_samples \\"
            echo "    --verify_bounds"
        done
    done
    exit 0
fi

# Run main function
main "$@"