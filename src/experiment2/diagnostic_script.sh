#!/bin/bash
# diagnostic_script.sh
# Complete diagnostic to identify pipeline issues

echo "========================================"
echo "Pipeline Diagnostic Script"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Configuration
BASE_DIR="/home/user/bilinear-projection-theory"
SISAP_DIR="/home/user/sisap2025"
CODE_DIR="$BASE_DIR/src/experiment2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Function to check if command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        print_status "OK" "$1 is available"
        return 0
    else
        print_status "ERROR" "$1 is not available"
        return 1
    fi
}

# Function to check directory
check_directory() {
    local dir=$1
    local description=$2
    if [ -d "$dir" ]; then
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        print_status "OK" "$description exists ($size)"
        return 0
    else
        print_status "ERROR" "$description does not exist: $dir"
        return 1
    fi
}

# Function to check file
check_file() {
    local file=$1
    local description=$2
    if [ -f "$file" ]; then
        local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "unknown")
        print_status "OK" "$description exists (${size} bytes)"
        return 0
    else
        print_status "ERROR" "$description does not exist: $file"
        return 1
    fi
}

echo "1. SYSTEM REQUIREMENTS CHECK"
echo "========================================"

# Check basic commands
check_command "python3"
check_command "pip"
check_command "git"

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version 2>&1)
    print_status "OK" "Python version: $python_version"
else
    print_status "ERROR" "Python3 not found"
fi

# Check disk space
echo ""
echo "Disk Space:"
df -h /home/user/ 2>/dev/null || df -h /home/ 2>/dev/null || echo "Could not check disk space"

echo ""
echo "2. PYTHON ENVIRONMENT CHECK"
echo "========================================"

# Check Python packages
packages=("torch" "transformers" "sentence_transformers" "ir_datasets" "pytrec_eval" "tqdm" "pandas" "matplotlib" "numpy")
for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
        print_status "OK" "$package ($version)"
    else
        print_status "ERROR" "$package not installed"
    fi
done

# Check CUDA
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    print_status "OK" "CUDA available (version: $cuda_version)"
else
    print_status "WARN" "CUDA not available (will use CPU)"
fi

echo ""
echo "3. DIRECTORY STRUCTURE CHECK"
echo "========================================"

# Check main directories
check_directory "$BASE_DIR" "Base project directory"
check_directory "$CODE_DIR" "Experiment2 code directory"
check_directory "$SISAP_DIR" "SISAP data directory"

# Check subdirectories
for subdir in "embeddings" "cv_triples" "results" "logs" "data"; do
    check_directory "$SISAP_DIR/$subdir" "SISAP $subdir directory"
done

# Check data subdirectories
for dataset in "car" "robust"; do
    check_directory "$SISAP_DIR/data/$dataset" "Dataset $dataset directory"
done

echo ""
echo "4. CODE FILES CHECK"
echo "========================================"

# Check essential Python files
essential_files=(
    "$CODE_DIR/config.py"
    "$CODE_DIR/models.py"
    "$CODE_DIR/train_cv.py"
    "$CODE_DIR/preprocess_embeddings.py"
    "$CODE_DIR/create_cv_triples.py"
    "$CODE_DIR/data_loader.py"
    "$CODE_DIR/evaluate.py"
    "$CODE_DIR/utils.py"
)

for file in "${essential_files[@]}"; do
    check_file "$file" "$(basename "$file")"
done

echo ""
echo "5. DATA FILES CHECK"
echo "========================================"

# Check data files for each dataset
for dataset in "car" "robust"; do
    dataset_dir="$SISAP_DIR/data/$dataset"
    echo "Dataset: $dataset"

    data_files=("queries.tsv" "qrels.txt" "run.txt" "folds.json")
    for file in "${data_files[@]}"; do
        check_file "$dataset_dir/$file" "  $dataset/$file"
    done
    echo ""
done

echo ""
echo "6. EMBEDDING FILES CHECK"
echo "========================================"

# Check for existing embeddings
embedding_models=("facebook-contriever" "microsoft-mpnet-base" "google-electra-base" "bert-base-uncased")
datasets=("car" "robust" "msmarco")

for model in "${embedding_models[@]}"; do
    model_dir="$SISAP_DIR/embeddings/$model"
    if [ -d "$model_dir" ]; then
        echo "Model: $model"
        for dataset in "${datasets[@]}"; do
            query_file="$model_dir/${dataset}_query_embeddings.npy"
            passage_file="$model_dir/${dataset}_passage_embeddings.npy"

            if [ -f "$query_file" ] && [ -f "$passage_file" ]; then
                print_status "OK" "  $dataset embeddings exist"
            else
                print_status "ERROR" "  $dataset embeddings missing"
            fi
        done
        echo ""
    else
        print_status "ERROR" "$model embedding directory missing"
    fi
done

echo ""
echo "7. CV TRIPLES CHECK"
echo "========================================"

for dataset in "car" "robust"; do
    triples_dir="$SISAP_DIR/cv_triples/$dataset"
    echo "Dataset: $dataset"

    if [ -d "$triples_dir" ]; then
        for fold in {0..4}; do
            triples_file="$triples_dir/fold_${fold}_triples.pt"
            test_file="$triples_dir/fold_${fold}_test_data.json"

            if [ -f "$triples_file" ] && [ -f "$test_file" ]; then
                print_status "OK" "  Fold $fold files exist"
            else
                print_status "ERROR" "  Fold $fold files missing"
            fi
        done
    else
        print_status "ERROR" "$dataset triples directory missing"
    fi
    echo ""
done

echo ""
echo "8. CONFIGURATION CHECK"
echo "========================================"

# Try to import and check config
if [ -f "$CODE_DIR/config.py" ]; then
    cd "$CODE_DIR"

    # Check if config loads without errors
    if python3 -c "import config; print('Config loaded successfully')" 2>/dev/null; then
        print_status "OK" "Config file loads without errors"

        # Check key configuration values
        device=$(python3 -c "import config; print(config.DEVICE)" 2>/dev/null || echo "unknown")
        embedding_dim=$(python3 -c "import config; print(config.EMBEDDING_DIM)" 2>/dev/null || echo "unknown")
        batch_size=$(python3 -c "import config; print(config.BATCH_SIZE)" 2>/dev/null || echo "unknown")

        echo "  Device: $device"
        echo "  Embedding dimension: $embedding_dim"
        echo "  Batch size: $batch_size"

    else
        print_status "ERROR" "Config file has import errors"
        python3 -c "import config" 2>&1 | head -5
    fi
else
    print_status "ERROR" "Config file not found"
fi

echo ""
echo "9. PERMISSIONS CHECK"
echo "========================================"

# Check write permissions
for dir in "$SISAP_DIR/embeddings" "$SISAP_DIR/cv_triples" "$SISAP_DIR/results" "$SISAP_DIR/logs"; do
    if [ -d "$dir" ]; then
        if [ -w "$dir" ]; then
            print_status "OK" "Write permission for $dir"
        else
            print_status "ERROR" "No write permission for $dir"
        fi
    fi
done

echo ""
echo "10. RECENT LOG FILES"
echo "========================================"

log_dir="$SISAP_DIR/logs"
if [ -d "$log_dir" ]; then
    recent_logs=$(find "$log_dir" -name "*.log" -mtime -1 2>/dev/null | head -5)
    if [ -n "$recent_logs" ]; then
        echo "Recent log files (last 24 hours):"
        echo "$recent_logs"
    else
        print_status "WARN" "No recent log files found"
    fi
else
    print_status "ERROR" "Log directory not found"
fi

echo ""
echo "========================================"
echo "DIAGNOSTIC SUMMARY"
echo "========================================"

# Count errors and warnings
error_count=$(grep -c "✗" /tmp/diagnostic_output.txt 2>/dev/null || echo "0")
warn_count=$(grep -c "⚠" /tmp/diagnostic_output.txt 2>/dev/null || echo "0")

echo "Completed at: $(date)"
echo ""

if [ "$error_count" -eq 0 ]; then
    print_status "OK" "No critical errors found! Your setup looks good."
    echo ""
    echo "NEXT STEPS:"
    echo "1. Run the embedding generation script if embeddings are missing"
    echo "2. Run the pipeline with: ./minimal_pipeline_fix.sh"
else
    print_status "ERROR" "$error_count critical errors found"
    echo ""
    echo "RECOMMENDED ACTIONS:"
    echo "1. Install missing Python packages with: pip install torch transformers sentence-transformers ir_datasets pytrec_eval"
    echo "2. Create missing directories with: mkdir -p /home/user/sisap2025/{embeddings,cv_triples,results,logs,data/{car,robust}}"
    echo "3. Download required datasets and place them in the data directories"
    echo "4. Run the setup script: ./setup_embeddings.sh"
    echo "5. Fix any configuration issues in config.py"
fi

if [ "$warn_count" -gt 0 ]; then
    print_status "WARN" "$warn_count warnings found (non-critical)"
fi

echo ""
echo "For detailed help, see the troubleshooting guide."
echo "========================================"