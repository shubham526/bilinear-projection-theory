# Experiment 2: MS MARCO Real-World Validation

This experiment validates our theoretical findings on a real-world information retrieval benchmark using MS MARCO V1 Passage Ranking dataset with fixed, pre-computed embeddings.

## 📋 Overview

This experiment demonstrates:
1. **Practical superiority** of bilinear models over dot-product baselines
2. **Effectiveness** of low-rank approximations (validating Theorem 4.1)
3. **Real-world applicability** of theoretical insights

## 🚀 Quick Start

```bash
cd experiment2/
python main_train.py  # After completing setup steps below
```

## 📁 Directory Structure

```
experiment2/
├── main_train.py              # Main training script
├── models.py                  # Model definitions (WDP, LRB, FRB, DotProduct)
├── data_loader.py             # Dataset and data loading utilities
├── config.py                  # Configuration file (paths, hyperparameters)
├── evaluate.py                # Evaluation functions
├── preprocess_embeddings.py   # Script to generate embeddings
├── requirements.txt           # Python dependencies for this experiment
├── embeddings/                # Directory for pre-computed embeddings (created)
├── data/                      # Directory for MS MARCO data (you create)
├── saved_models/              # Directory for saved models and results (created)
└── ms_marco_eval/             # Directory for MS MARCO evaluation scripts
```

## ⚙️ Setup Instructions

### Step 1: Install Dependencies

```bash
# From the experiment2/ directory
pip install -r requirements.txt
```

### Step 2: Download MS MARCO V1 Data

Create the data directory and download required files:

```bash
mkdir -p data/msmarco_v1/
cd data/msmarco_v1/

# Download the MS MARCO passage collection
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzf collection.tar.gz

# Download training triples
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
tar -xzf triples.train.small.tar.gz

# Download dev queries and qrels
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tar.gz
tar -xzf queries.dev.small.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tar.gz
tar -xzf qrels.dev.small.tar.gz

# Download top1000 candidates for dev set (for faster evaluation)
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
tar -xzf top1000.dev.tar.gz

cd ../..  # Return to experiment2/
```

### Step 3: Download MS MARCO Evaluation Script

```bash
mkdir -p ms_marco_eval/
cd ms_marco_eval/
wget https://raw.githubusercontent.com/microsoft/MSMARCO-Passage-Ranking/master/ms_marco_eval.py
mv ms_marco_eval.py msmarco_passage_eval.py
cd ..
```

### Step 4: Configure Paths

Edit `config.py` to set correct paths for your setup:

```python
# Update these paths to match your directory structure
MSMARCO_V1_DIR = "data/msmarco_v1/"
EMBEDDING_DIR = "embeddings/sbert_all-mpnet-base-v2/"
MSMARCO_EVAL_SCRIPT = "ms_marco_eval/msmarco_passage_eval.py"
```

### Step 5: Generate Embeddings

**⚠️ IMPORTANT**: This step is required before training and can take several hours.

```bash
# Basic usage
python preprocess_embeddings.py

# With options
python preprocess_embeddings.py --skip-if-exists  # Skip if embeddings exist
python preprocess_embeddings.py --verify-only     # Just verify existing embeddings
python preprocess_embeddings.py --model-name sentence-transformers/all-MiniLM-L6-v2  # Different model
```

This will:
- Extract all unique query and passage IDs from training and dev data
- Load corresponding text from MS MARCO files
- Generate SBERT embeddings using `all-mpnet-base-v2` model (768-dim)
- Save embeddings and ID mappings to `embeddings/` directory

**Expected time**: 2-6 hours depending on hardware and chosen model

## 🏃‍♂️ Running the Experiment

### Quick Test Run

For initial testing with a small subset:

```python
# Edit main_train.py and set:
train_dataset_limit = 10000  # Use only 10k training triples

# In config.py, test fewer models:
models_to_run = ["dot_product", "weighted_dot_product", "low_rank_bilinear_32"]
```

Then run:
```bash
python main_train.py
```

### Full Training

```bash
python main_train.py
```

This will:
1. Train all models defined in `config.MODEL_CONFIGS`
2. Evaluate on dev set after each epoch  
3. Save the best model for each configuration
4. Log training progress and results

**Expected time**: 8-12 hours for all models

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
├── {model_name}/
│   ├── model.pt                   # Trained model weights
│   ├── eval_results.txt          # Human-readable results
│   ├── results.json              # Structured results
│   └── training.log              # Detailed training logs
└── summary_results.json          # Comparative summary
```

### Key Metrics

- **MRR@10**: Primary metric for MS MARCO
- **Recall@100, Recall@1000**: Coverage metrics
- **Training time**: Efficiency comparison
- **Parameter count**: Model size comparison

### Expected Results

Based on our experiments:

| Model | MRR@10 | Improvement | Parameters |
|-------|--------|-------------|------------|
| Dot Product | ~0.190 | Baseline | 0 |
| Weighted DP | ~0.195 | +2.6% | 768 |
| LRB (rank=32) | ~0.205 | +7.9% | 49K |
| LRB (rank=64) | ~0.215 | +13.2% | 98K |
| LRB (rank=128) | ~0.218 | +14.7% | 196K |
| Full Bilinear | ~0.220 | +15.8% | 590K |

## 🔧 Configuration Options

### Model Selection

In `config.py`, choose which models to train:

```python
# Train all models (default)
models_to_run = ["dot_product", "weighted_dot_product", "low_rank_bilinear_32", 
                 "low_rank_bilinear_64", "low_rank_bilinear_128", "full_rank_bilinear"]

# Train only efficient models
models_to_run = ["dot_product", "weighted_dot_product", "low_rank_bilinear_64"]

# Train only bilinear models
models_to_run = ["low_rank_bilinear_32", "low_rank_bilinear_64", "low_rank_bilinear_128"]
```

### Hyperparameter Tuning

Key parameters in `config.py`:

```python
# Training parameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 1024
NUM_EPOCHS = 10
MARGIN = 0.01  # Margin for ranking loss

# Model-specific parameters
EMBEDDING_DIM = 768  # Match your SBERT model
LOW_RANK_DIMS = [32, 64, 128]  # Ranks to test for LRB models

# Data parameters
TRAIN_DATASET_LIMIT = None  # Set to integer for subset training
```

### Different SBERT Models

To use a different SBERT model:

1. Update `config.py`:
```python
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Update to match model dimension
```

2. Regenerate embeddings:
```bash
python preprocess_embeddings.py
```

## 🐛 Troubleshooting

### Common Issues

**1. FileNotFoundError during embedding loading**
```
Solution: Ensure preprocess_embeddings.py completed successfully
Check: ls embeddings/ should show .npy files and mappings
```

**2. CUDA out of memory**
```
Solution 1: Reduce BATCH_SIZE in config.py (try 512 or 256)
Solution 2: Use a smaller SBERT model for preprocessing
Solution 3: Set DEVICE="cpu" in config.py (slower but works)
```

**3. Evaluation script errors**
```
Solution: Verify ms_marco_eval/msmarco_passage_eval.py exists
Check: Ensure MSMARCO_EVAL_SCRIPT path is correct in config.py
```

**4. Slow preprocessing**
```
Expected: Preprocessing is CPU/GPU intensive and takes time
Tip: Use --skip-if-exists flag if embeddings already exist
Option: Start with smaller model like all-MiniLM-L6-v2
```

### Debugging Steps

1. **Verify embeddings**:
```bash
python preprocess_embeddings.py --verify-only
```

2. **Test with small dataset**:
```python
# In main_train.py, set:
train_dataset_limit = 1000
```

3. **Check GPU usage**:
```bash
nvidia-smi  # Monitor GPU memory
```

## 🚀 Advanced Usage

### Custom Models

To add a new model:

1. Add model class to `models.py`:
```python
class MyCustomModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # Define your model
    
    def forward(self, query_embeds, passage_embeds):
        # Implement forward pass
        return scores
```

2. Add configuration to `config.py`:
```python
MODEL_CONFIGS["my_custom_model"] = {
    "class": MyCustomModel,
    "params": {"embedding_dim": EMBEDDING_DIM},
    "lr": 1e-5
}
```

3. Add to training list:
```python
models_to_run.append("my_custom_model")
```

### Batch Processing

For large-scale evaluation:

```python
# In evaluate.py, modify eval_batch_size for memory optimization
eval_batch_size = 256  # Reduce if memory issues
```

### Export for Production

After training, export optimal models:

```python
import torch

# Load best LRB model
model = torch.load("saved_models/low_rank_bilinear_64/model.pt")
model.eval()

# Save for production
torch.jit.save(torch.jit.script(model), "production_model.pt")
```

## 📈 Analysis Scripts

Create analysis scripts for deeper insights:

```python
# analyze_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load all results
with open("saved_models/summary_results.json", "r") as f:
    results = json.load(f)

# Create comparison plots
models = list(results.keys())
mrr_scores = [results[m]["mrr_at_10"] for m in models]
params = [results[m]["num_parameters"] for m in models]

# Plot accuracy vs efficiency
plt.figure(figsize=(10, 6))
plt.scatter(params, mrr_scores)
for i, model in enumerate(models):
    plt.annotate(model, (params[i], mrr_scores[i]))
plt.xlabel("Number of Parameters")
plt.ylabel("MRR@10")
plt.title("Model Efficiency vs Performance")
plt.show()
```

## ✅ Validation Checklist

Before considering the experiment complete:

- [ ] All models trained successfully
- [ ] Evaluation completes without errors
- [ ] Results show expected relative performance (LRB > WDP > DP)
- [ ] Low-rank models achieve 90%+ of full-rank performance
- [ ] Results saved in all expected formats

## 📚 Related Resources

- **MS MARCO Homepage**: https://microsoft.github.io/msmarco/
- **SBERT Documentation**: https://www.sbert.net/
- **Paper Preprint**: [Link when available]
- **Experiment 1**: See `../README_experiment1.md` for synthetic validation

---

**Next Steps**: After completing this experiment, compare results with theoretical predictions and consider running analysis scripts to generate publication-ready figures.