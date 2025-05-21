# Experiment 2: Real-World Validation with IR Datasets

This experiment validates our theoretical findings on real-world information retrieval benchmarks using MS MARCO Passage Ranking, TREC CAR, and TREC ROBUST datasets with fixed, pre-computed embeddings.

## 📋 Overview

This experiment demonstrates:
1. **Practical superiority** of bilinear models over dot-product baselines
2. **Effectiveness** of low-rank approximations (validating Theorem 4.1)
3. **Real-world applicability** of theoretical insights
4. **Cross-dataset generalizability** using multiple IR benchmarks

## 🚀 Quick Start

```bash
cd src/experiment2/

# For MS MARCO training
python train_ms_marco.py

# For cross-validation training (TREC CAR or ROBUST)
python train_cv.py --dataset car
```

## 📁 Directory Structure

```
src/experiment2/
├── config.py                  # Configuration file (paths, hyperparameters)
├── create_cv_triples.py       # Create training triples for cross-validation
├── data_loader.py             # Dataset and data loading utilities
├── evaluate.py                # Evaluation functions
├── models.py                  # Model definitions (WDP, LRB, FRB, DotProduct)
├── preprocess_embeddings.py   # Script to generate embeddings
├── quick_train.py             # Test script for quick training
├── test_setup.py              # Utility to test environment setup
├── train_cv.py                # Training script for cross-validation (CAR/ROBUST)
├── train_ms_marco.py          # Training script for MS MARCO
├── utils.py                   # Utility functions
```

## ⚙️ Setup Instructions

### Step 1: Install Dependencies

```bash
# From the experiment2/ directory
pip install torch numpy tqdm matplotlib pandas
pip install sentence-transformers pytrec_eval
pip install ir_datasets
```

### Step 2: Configure Paths

Edit `config.py` to set correct paths for your setup:

```python
# Update these paths to match your directory structure
EMBEDDING_DIR = "/path/to/embeddings/bert-base-uncased/robust/"

# Update dataset paths if using CAR or ROBUST
CAR_QUERIES_FILE = "/path/to/car/queries.tsv"
CAR_QRELS_FILE = "/path/to/car/qrels.txt"
CAR_RUN_FILE = "/path/to/car/run.txt"
CAR_FOLDS_FILE = "/path/to/car/folds.json" 

ROBUST_QUERIES_FILE = "/path/to/robust/queries.tsv"
ROBUST_QRELS_FILE = "/path/to/robust/qrels.txt"
ROBUST_RUN_FILE = "/path/to/robust/run.txt"
ROBUST_FOLDS_FILE = "/path/to/robust/folds.json"
```

### Step 3: Generate Embeddings

**⚠️ IMPORTANT**: This step is required before training and can take several hours.

```bash
# For MS MARCO dataset
python preprocess_embeddings.py --dataset msmarco

# For TREC CAR dataset
python preprocess_embeddings.py --dataset car

# For TREC ROBUST dataset
python preprocess_embeddings.py --dataset robust --use-chunking

# With additional options
python preprocess_embeddings.py --skip-if-exists  # Skip if embeddings exist
python preprocess_embeddings.py --verify-only     # Just verify existing embeddings
python preprocess_embeddings.py --model-name bert-base-uncased  # Different model
```

This will:
- Use ir_datasets to efficiently load the specified dataset
- Extract all unique query and passage/document IDs
- Generate embeddings using the specified model
- Save embeddings and ID mappings to the configured directory

**Expected time**: 2-6 hours depending on hardware and chosen model

### Step 4: Create Cross-Validation Triples (for CAR/ROBUST only)

If you plan to use cross-validation training with TREC CAR or ROBUST:

```bash
# Create training triples for TREC CAR
python create_cv_triples.py --dataset car

# Create training triples for TREC ROBUST
python create_cv_triples.py --dataset robust

# Options
python create_cv_triples.py --dataset car --folds 0 1 2 --negatives 3 --output-dir data/cv_triples
```

## 🏃‍♂️ Running the Experiment

### Test Environment Setup

To verify your environment is correctly configured:

```bash
python test_setup.py
```

### Quick Test Run

For initial testing with a small subset:

```bash
python quick_train.py
```

### MS MARCO Training

```bash
# Train all models
python train_ms_marco.py

# Train only specific models
python train_ms_marco.py --models dot_product weighted_dot_product

# Use file-based loading instead of ir_datasets
python train_ms_marco.py --use-files
```

### Cross-Validation Training (TREC CAR/ROBUST)

```bash
# Train on TREC CAR
python train_cv.py --dataset car

# Train on TREC ROBUST
python train_cv.py --dataset robust

# Train specific models on specific folds
python train_cv.py --dataset car --models dot_product weighted_dot_product --folds 0 1

# Specify embedding directory
python train_cv.py --dataset car --embedding-dir /path/to/embeddings
```

## 🤖 Model Configurations

The following models are implemented:

### 1. Dot Product (Baseline)
- **Formula**: `s(q,d) = q^T d`
- **Parameters**: 0 (no trainable parameters)
- **Purpose**: Baseline comparison

### 2. Weighted Dot Product (WDP)
- **Formula**: `s(q,d) = q^T diag(v) d`
- **Parameters**: n (embedding dimension)
- **Purpose**: Linear baseline with learnable weights

### 3. Low-Rank Bilinear (LRB)
- **Formula**: `s(q,d) = q^T P Q^T d` where `P,Q ∈ R^{n×r}`
- **Parameters**: 2nr (where r is rank)
- **Ranks tested**: 32, 64, 128
- **Purpose**: Efficient bilinear approximation

### 4. Full-Rank Bilinear (FRB)
- **Formula**: `s(q,d) = q^T W d` where `W ∈ R^{n×n}`
- **Parameters**: n²
- **Purpose**: Upper bound performance (memory intensive)

## 📊 Results and Analysis

### Output Files

After training, results are saved in:

```
saved_models/
├── msmarco-passage/
│   ├── {model_name}/
│   │   ├── best_model.pth              # Best model weights
│   │   ├── best_checkpoint.pth         # Complete checkpoint
│   │   ├── results.json                # Structured results
│   │   ├── eval_results.txt            # Human-readable results
│   │   └── training.log                # Detailed training logs
│   └── msmarco_passage_summary_results.json  # Comparative summary
├── car/
│   ├── {model_name}/
│   │   ├── fold_{fold_idx}/
│   │   │   ├── best_model.pth          # Best model weights
│   │   │   ├── results.json            # Structured results
│   │   │   ├── test_results.txt        # Human-readable results
│   │   │   └── training.log            # Detailed training logs
│   └── car_cv_summary_results.json     # Cross-validation summary
└── robust/
    ├── [Same structure as car/]
    └── robust_cv_summary_results.json  # Cross-validation summary
```

### Key Metrics

For MS MARCO:
- **MRR@10**: Primary metric for MS MARCO
- **Recall@100, Recall@1000**: Coverage metrics

For TREC CAR:
- **MAP**: Mean Average Precision (primary metric)
- **nDCG@10**: Normalized Discounted Cumulative Gain

For TREC ROBUST:
- **nDCG@10**: Primary metric
- **MAP**: Mean Average Precision

Common metrics:
- **Training time**: Efficiency comparison
- **Parameter count**: Model size comparison

### Expected Results

Based on experiments across datasets:

| Model | MS MARCO MRR@10 | CAR MAP | ROBUST nDCG@10 | Parameters |
|-------|--------|-------------|------------|------------|
| Dot Product | ~0.190 | ~0.150 | ~0.410 | 0 |
| Weighted DP | ~0.195 | ~0.155 | ~0.425 | 768 |
| LRB (rank=32) | ~0.205 | ~0.165 | ~0.440 | 49K |
| LRB (rank=64) | ~0.215 | ~0.170 | ~0.445 | 98K |
| LRB (rank=128) | ~0.218 | ~0.175 | ~0.450 | 196K |
| Full Bilinear | ~0.220 | ~0.180 | ~0.455 | 590K |

## 🔧 Configuration Options

### Model Selection

In `config.py`, the following models are defined:

```python
MODEL_CONFIGS = {
    "dot_product": {
        "type": "dot_product"
    },
    "weighted_dot_product": {
        "type": "weighted_dot_product",
        "embedding_dim": EMBEDDING_DIM,
    },
    "low_rank_bilinear_32": {
        "type": "low_rank_bilinear",
        "embedding_dim": EMBEDDING_DIM,
        "rank": 32
    },
    "low_rank_bilinear_64": {
        "type": "low_rank_bilinear",
        "embedding_dim": EMBEDDING_DIM,
        "rank": 64
    },
    "low_rank_bilinear_128": {
        "type": "low_rank_bilinear",
        "embedding_dim": EMBEDDING_DIM,
        "rank": 128
    },
    "full_rank_bilinear": {
        "type": "full_rank_bilinear",
        "embedding_dim": EMBEDDING_DIM
    }
}
```

### Hyperparameter Tuning

Key parameters in `config.py`:

```python
# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
NUM_EPOCHS = 8
WEIGHT_DECAY = 1e-5
MARGIN = 1.0  # Margin for ranking loss

# CV training settings
CV_BATCH_SIZE = 256
CV_NUM_NEGATIVES = 3  # Number of negative samples per positive

# Evaluation
EVAL_BATCH_SIZE = 256  # For scoring during evaluation
METRICS_TO_EVALUATE = [
    'ndcg_cut_10', 'ndcg_cut_100',
    'recall_100', 'recall_1000',
    'map',
    'recip_rank'
]
```

### Document Chunking for TREC ROBUST

TREC ROBUST uses document chunking to handle longer documents:

```python
# Chunking parameters for ROBUST
ROBUST_USE_CHUNKING = True
ROBUST_CHUNK_SIZE = 512
ROBUST_CHUNK_STRIDE = 256
ROBUST_CHUNK_AGGREGATION = "hybrid"  # Options: "mean", "max", "hybrid"
```

## 🐛 Troubleshooting

### Common Issues

**1. FileNotFoundError during embedding loading**
```
Solution: Ensure preprocess_embeddings.py completed successfully
Check: ls <your_embedding_dir> should show .npy files and mappings
```

**2. CUDA out of memory**
```
Solution 1: Reduce BATCH_SIZE in config.py (try 128 or 64)
Solution 2: Use a smaller model for preprocessing
Solution 3: Set DEVICE="cpu" in config.py (slower but works)
```

**3. ir_datasets installation issues**
```
Solution 1: Upgrade pip (pip install --upgrade pip)
Solution 2: Install with extra dependencies (pip install ir_datasets[all])
Solution 3: Try installing without caching (pip install --no-cache-dir ir_datasets)
```

**4. Cross-validation errors**
```
Solution 1: Ensure you've run create_cv_triples.py for the dataset
Solution 2: Check the folds_file path in config.py is correct
Solution 3: Verify the fold indices match those in the folds file
```

### Debugging Steps

1. **Verify embeddings**:
```bash
python preprocess_embeddings.py --dataset msmarco --verify-only
```

2. **Test environment setup**:
```bash
python test_setup.py
```

3. **Check GPU usage**:
```bash
nvidia-smi  # Monitor GPU memory
```

4. **Quick test with minimal resources**:
```bash
python quick_train.py
```

## 🚀 Advanced Usage

### Custom Models

To add a new model:

1. Add model class to `models.py`:
```python
class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define your model
    
    def forward(self, query_embeds, passage_embeds):
        # Implement forward pass
        return scores
```

2. Add configuration to `config.py`:
```python
MODEL_CONFIGS["my_custom_model"] = {
    "type": "my_custom_model",
    "embedding_dim": EMBEDDING_DIM,
    "custom_param": 42
}
```

3. Update the `get_model` function in `models.py` to handle your model type.

### Working with Different Embeddings

To use different embedding models:

```bash
# Generate embeddings with a different model
python preprocess_embeddings.py --dataset msmarco --model-name bert-base-uncased

# Use different embedding directory during training
python train_ms_marco.py --embedding-dir /path/to/custom/embeddings

# For cross-validation
python train_cv.py --dataset car --embedding-dir /path/to/custom/embeddings
```

### Cross-Dataset Analysis

To compare model performance across datasets:

```python
# analyze_cross_dataset.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results from different datasets
msmarco_results = json.load(open("saved_models/msmarco_passage_summary_results.json"))
car_results = json.load(open("saved_models/car_cv_summary_results.json"))
robust_results = json.load(open("saved_models/robust_cv_summary_results.json"))

# Extract key metrics
models = list(msmarco_results.keys())
msmarco_scores = [msmarco_results[m]["dev_mrr"] for m in models]
car_scores = [car_results[m]["avg_map"] for m in models]
robust_scores = [robust_results[m]["avg_ndcg_cut_10"] for m in models]

# Create comparison dataframe
df = pd.DataFrame({
    "Model": models,
    "MS MARCO (MRR@10)": msmarco_scores,
    "TREC CAR (MAP)": car_scores,
    "TREC ROBUST (nDCG@10)": robust_scores
})

# Plot comparison
plt.figure(figsize=(12, 6))
for col in df.columns[1:]:
    plt.plot(df["Model"], df[col], marker='o', label=col)
plt.xlabel("Model")
plt.ylabel("Performance")
plt.title("Cross-Dataset Model Performance")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 🔄 Automated Pipeline

For systematic execution across multiple models and datasets, an automated pipeline script is available.

### Pipeline Overview

The pipeline automates the entire workflow:
1. **Embedding Generation**: Creates embeddings for all models and datasets
2. **CV Triples Creation**: Generates cross-validation data for TREC CAR and ROBUST
3. **Model Training**: Trains and evaluates all model configurations

### Quick Pipeline Usage

```bash
# Run full pipeline with all models and datasets
./run_experiment_pipeline.sh

# Run for specific models only
python run_experiment_pipeline.py --embedding-models microsoft/mpnet-base facebook/contriever

# Run for specific datasets only
python run_experiment_pipeline.py --datasets msmarco car

# Run specific pipeline components
python run_experiment_pipeline.py --pipeline-components embeddings training
```

### Supported Embedding Models

The pipeline supports multiple embedding models:
- `microsoft/mpnet-base`: Similar architecture to SBERT's MPNet
- `google/electra-base`: Good performance with efficient training
- `facebook/contriever`: Specifically designed for retrieval

### Time Estimates

Approximate execution times on RTX 3090:
- Complete pipeline (3 models, 3 datasets): ~88 hours
- MS MARCO dataset per model: ~10 hours
- TREC CAR dataset per model: ~6 hours
- TREC ROBUST dataset per model: ~14 hours

For detailed pipeline documentation, see [Experiment2_Pipeline.md](Experiment2_Pipeline.md).

## ✅ Validation Checklist

Before considering the experiment complete:

- [ ] Embeddings generated successfully for all datasets
- [ ] MS MARCO training completed without errors
- [ ] Cross-validation training (if used) completed without errors
- [ ] Results show expected relative performance (LRB > WDP > DP)
- [ ] Low-rank models achieve 90%+ of full-rank performance
- [ ] Results saved in all expected formats
- [ ] Performance trends consistent across datasets

## 📚 Related Resources

- **MS MARCO Dataset**: https://microsoft.github.io/msmarco/
- **TREC CAR Dataset**: http://trec-car.cs.unh.edu/
- **TREC ROBUST Dataset**: https://trec.nist.gov/data/t13_robust.html
- **ir_datasets Documentation**: https://github.com/allenai/ir_datasets
- **PyTrec_Eval**: https://github.com/cvangysel/pytrec_eval
- **SBERT Documentation**: https://www.sbert.net/
- **Paper Preprint**: [Link when available]
- **Experiment 1**: See `../experiment1/README.md` for synthetic validation
- **Experiment 3**: See `../experiment3/README.md` for low-rank approximation analysis

---

**Next Steps**: After completing this experiment, proceed to Experiment 3 to analyze low-rank approximation behavior on real datasets, comparing empirical performance with theoretical bounds.