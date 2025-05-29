#!/bin/bash

# experiment3_batch_runner.sh - CV fold aware version
set -e

# Configuration
BASE_DIR="/home/user/sisap2025"
RESULTS_BASE_DIR="$BASE_DIR/results"
EXP2_RESULTS_DIR="$RESULTS_BASE_DIR/experiment2"
EXP3_RESULTS_DIR="$RESULTS_BASE_DIR/experiment3"
SCRIPT_PATH="/home/user/bilinear-projection-theory/src/experiment3/experiment3.py"

# Arrays for datasets and models
datasets=("car" "robust" "msmarco")
models=("bert-base-uncased" "facebook-contriever" "microsoft-mpnet-base")

# Experiment parameters
ranks=(1 2 4 8 16 32 64 128)
device="cuda"
num_samples=100000

# CV folds (for car and robust)
cv_folds=(0 1 2 3 4)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to find model files for a dataset/model combination
find_model_files() {
    local dataset="$1"
    local model="$2"
    local model_files=()

    if [[ "$dataset" == "msmarco" ]]; then
        # MS MARCO doesn't use CV, look for single model
        local single_model_paths=(
            "$EXP2_RESULTS_DIR/$model/$dataset/full_rank_bilinear/best_model.pth"
            "$EXP2_RESULTS_DIR/$dataset/$model/full_rank_bilinear/best_model.pth"
        )

        for path in "${single_model_paths[@]}"; do
            if [[ -f "$path" ]]; then
                model_files+=("$path")
                break
            fi
        done
    else
        # CAR and ROBUST use 5-fold CV
        for fold in "${cv_folds[@]}"; do
            local fold_paths=(
                "$EXP2_RESULTS_DIR/$model/$dataset/full_rank_bilinear_fold$fold/best_model.pth"
                "$EXP2_RESULTS_DIR/$dataset/$model/full_rank_bilinear_fold$fold/best_model.pth"
            )

            for path in "${fold_paths[@]}"; do
                if [[ -f "$path" ]]; then
                    model_files+=("$path")
                    break
                fi
            done
        done
    fi

    printf '%s\n' "${model_files[@]}"
}

# Function to run experiment on a single model
run_single_experiment() {
    local dataset="$1"
    local model="$2"
    local model_path="$3"
    local fold_id="$4"
    local output_dir="$5"

    local fold_suffix=""
    if [[ -n "$fold_id" ]]; then
        fold_suffix="_fold$fold_id"
    fi

    log_info "🚀 Running experiment: Dataset=$dataset, Model=$model$fold_suffix"

    local cmd="python \"$SCRIPT_PATH\" \
        --model-path \"$model_path\" \
        --dataset \"$dataset\" \
        --ranks ${ranks[*]} \
        --results-dir \"$output_dir\" \
        --device \"$device\" \
        --num-samples $num_samples \
        --verify-bounds"

    if eval "$cmd"; then
        log_info "✅ Experiment completed: Dataset=$dataset, Model=$model$fold_suffix"
        return 0
    else
        log_error "❌ Experiment failed: Dataset=$dataset, Model=$model$fold_suffix"
        return 1
    fi
}

# Function to aggregate CV results
aggregate_cv_results() {
    local dataset="$1"
    local model="$2"
    local base_output_dir="$3"

    log_info "📊 Aggregating CV results for $dataset/$model"

    # Create aggregation script
    cat > "/tmp/aggregate_cv.py" << 'EOF'
import json
import numpy as np
import sys
import os

def aggregate_cv_results(base_dir, dataset, model):
    fold_results = []

    for fold in range(5):
        fold_dir = f"{base_dir}/fold_{fold}"
        summary_file = f"{fold_dir}/experiment3_summary.json"

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
                fold_results.append(data)

    if not fold_results:
        print(f"No valid fold results found for {dataset}/{model}")
        return

    # Aggregate scores across folds
    all_ranks = fold_results[0]['ranks_evaluated']
    aggregated_scores = []

    for i, rank in enumerate(all_ranks):
        rank_scores = [result['scores'][i] for result in fold_results if i < len(result['scores'])]
        if rank_scores:
            aggregated_scores.append({
                'rank': rank,
                'mean_score': np.mean(rank_scores),
                'std_score': np.std(rank_scores),
                'scores_per_fold': rank_scores
            })

    # Create aggregated summary
    aggregated_summary = {
        'dataset': dataset,
        'model': model,
        'num_folds': len(fold_results),
        'primary_metric': fold_results[0]['primary_metric'],
        'aggregated_results': aggregated_scores,
        'best_rank_mean': max(aggregated_scores, key=lambda x: x['mean_score'])['rank'],
        'best_score_mean': max(aggregated_scores, key=lambda x: x['mean_score'])['mean_score'],
        'individual_fold_results': fold_results
    }

    # Save aggregated results
    output_file = f"{base_dir}/cv_aggregated_summary.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated_summary, f, indent=2)

    print(f"Aggregated results saved to: {output_file}")
    print(f"Best rank (mean): {aggregated_summary['best_rank_mean']}")
    print(f"Best score (mean): {aggregated_summary['best_score_mean']:.4f}")

if __name__ == "__main__":
    aggregate_cv_results(sys.argv[1], sys.argv[2], sys.argv[3])
EOF

    python /tmp/aggregate_cv.py "$base_output_dir" "$dataset" "$model"
}

# Main execution
main() {
    log_info "Starting Experiment 3 batch runner (CV-aware)"

    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    local skipped_experiments=0

    for dataset in "${datasets[@]}"; do
        log_info "Processing dataset: $dataset"

        for model in "${models[@]}"; do
            log_info "Processing model: $model for dataset: $dataset"

            # Find all model files for this dataset/model combination
            readarray -t model_files < <(find_model_files "$dataset" "$model")

            if [[ ${#model_files[@]} -eq 0 ]]; then
                log_error "❌ No model files found for dataset=$dataset, model=$model"
                skipped_experiments=$((skipped_experiments + 1))
                continue
            fi

            log_info "Found ${#model_files[@]} model file(s) for $dataset/$model"

            # Create base output directory
            base_output_dir="$EXP3_RESULTS_DIR/$dataset/$model"

            local fold_success_count=0

            # Process each model file
            for i in "${!model_files[@]}"; do
                model_path="${model_files[$i]}"
                total_experiments=$((total_experiments + 1))

                # Determine fold ID and output directory
                if [[ "$dataset" == "msmarco" ]]; then
                    fold_id=""
                    output_dir="$base_output_dir"
                else
                    fold_id="$i"
                    output_dir="$base_output_dir/fold_$i"
                fi

                mkdir -p "$output_dir"

                # Check if experiment already completed
                if [[ -f "$output_dir/experiment3_summary.json" ]]; then
                    log_info "⏭️  Experiment already completed, skipping: $dataset/$model/fold_$i"
                    skipped_experiments=$((skipped_experiments + 1))
                    fold_success_count=$((fold_success_count + 1))
                    continue
                fi

                # Run the experiment
                if run_single_experiment "$dataset" "$model" "$model_path" "$fold_id" "$output_dir"; then
                    successful_experiments=$((successful_experiments + 1))
                    fold_success_count=$((fold_success_count + 1))
                else
                    failed_experiments=$((failed_experiments + 1))
                fi

                sleep 2
            done

            # Aggregate results if we have CV folds
            if [[ "$dataset" != "msmarco" && $fold_success_count -gt 1 ]]; then
                aggregate_cv_results "$dataset" "$model" "$base_output_dir"
            fi
        done
    done

    # Final summary
    log_info "Batch run completed!"
    log_info "Statistics:"
    log_info "  Total experiments: $total_experiments"
    log_info "  Successful: $successful_experiments"
    log_info "  Failed: $failed_experiments"
    log_info "  Skipped: $skipped_experiments"
}

# Handle command line arguments
case "${1:-}" in
    --dry-run)
        log_info "DRY RUN MODE - Would execute experiments for:"
        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do
                readarray -t model_files < <(find_model_files "$dataset" "$model")
                if [[ ${#model_files[@]} -gt 0 ]]; then
                    log_info "✅ $dataset/$model -> ${#model_files[@]} file(s)"
                    for file in "${model_files[@]}"; do
                        log_info "    $file"
                    done
                else
                    log_warn "❌ $dataset/$model -> NOT FOUND"
                fi
            done
        done
        exit 0
        ;;
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --dry-run      Show what would be executed"
        echo "  --help         Show this help"
        echo ""
        echo "This script handles both single models (MS MARCO) and 5-fold CV models (CAR, ROBUST)."
        exit 0
        ;;
esac

main "$@"