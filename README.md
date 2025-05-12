# Empirical Validation of Theoretical Advantages of Bilinear Similarities in Dense Retrieval

This repository contains the code and experimental setup for the paper titled "On the Theoretical Advantages of Bilinear Similarities in Dense Information Retrieval." The paper presents a comprehensive theoretical analysis establishing that bilinear similarity functions ($s(q,d) = q^T W d$) offer fundamental expressiveness advantages over standard dot-product and weighted dot-product similarities when query and document embeddings are **fixed**.

Our theoretical contributions are threefold:
1.  **Enhanced Expressiveness (Theorem 2.1):** We prove that bilinear similarities can capture strictly more ranking patterns than dot-product similarities under fixed embeddings.
2.  **Separation via Structured Task (Theorem 3.1):** We introduce the "Structured Agreement Ranking Task" where simple rank-2 bilinear models achieve perfect performance, while all weighted dot-product models are proven to fail universally. This highlights scenarios where modeling feature interactions is essential.
3.  **Low-Rank Approximation Bounds (Theorem 4.1):** We derive tight pointwise error bounds for low-rank approximations of bilinear matrices ($W_r$), showing that approximation quality is controlled by neglected singular values, providing a principled way to trade efficiency for accuracy.

This codebase provides the tools to empirically validate these theoretical findings. The experiments focus on:
* **Directly testing the theoretical predictions** using synthetic data, particularly for the Structured Agreement Ranking Task.
* **Evaluating the practical performance** of dot-product, weighted dot-product, and various bilinear similarity models (full-rank and low-rank) on a standard large-scale information retrieval benchmark (MS MARCO V1 Passage Ranking) using **fixed, pre-computed text embeddings**.
* **Analyzing the behavior of low-rank bilinear approximations** in relation to their theoretical error bounds and practical retrieval effectiveness.

The core idea is to learn the parameters of the similarity function ($W$ for bilinear, $v$ for weighted dot-product) on top of static embeddings, isolating the representational power of the similarity function itself.

## Project Structure

```
bilinear-proj-theory/
├── main_train.py          # Main training script
├── models.py              # Model definitions (WDP, LRB, FRB, DotProduct)
├── data_loader.py         # Dataset and data loading utilities
├── config.py              # Configuration file (paths, hyperparameters)
├── evaluate.py            # Evaluation functions
├── preprocess_embeddings.py  # Script to generate embeddings
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── embeddings/            # Directory for pre-computed embeddings (created)
├── data/                  # Directory for MS MARCO data (you create)
├── saved_models/          # Directory for saved models (created)
└── ms_marco_eval/         # Directory for MS MARCO evaluation script
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MS MARCO V1 Data

Create a `data/msmarco_v1/` directory and download the following files:

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
```

### 3. Download MS MARCO Evaluation Script

```bash
mkdir -p ms_marco_eval/
cd ms_marco_eval/
wget https://raw.githubusercontent.com/microsoft/MSMARCO-Passage-Ranking/master/ms_marco_eval.py
mv ms_marco_eval.py msmarco_passage_eval.py
cd ..
```

### 4. Update Configuration

Edit `config.py` to set the correct paths for your setup:

```python
# Update these paths to match your directory structure
MSMARCO_V1_DIR = "data/msmarco_v1/"
EMBEDDING_DIR = "embeddings/sbert_all-mpnet-base-v2/"
MSMARCO_EVAL_SCRIPT = "ms_marco_eval/msmarco_passage_eval.py"
```

### 5. Generate Embeddings

**Important**: This step is required before training and can take several hours.

```bash
python preprocess_embeddings.py
```

This will:
- Extract all unique query and passage IDs from the training and dev data
- Load the corresponding text from the MS MARCO files
- Generate SBERT embeddings using the `all-mpnet-base-v2` model
- Save embeddings and ID mappings to the `embeddings/` directory

You can monitor progress and use additional options:

```bash
# Skip if embeddings already exist
python preprocess_embeddings.py --skip-if-exists

# Verify existing embeddings
python preprocess_embeddings.py --verify-only

# Use a different SBERT model
python preprocess_embeddings.py --model-name sentence-transformers/all-MiniLM-L6-v2
```

## Running Training

### Quick Test Run

For initial testing with a small subset of data:

```python
# Edit main_train.py and set:
train_dataset_limit = 10000  # Use only 10k training triples

# In config.py, comment out models you don't want to test:
models_to_run = ["dot_product", "weighted_dot_product"]
```

### Full Training

```bash
python main_train.py
```

This will:
- Train all models defined in `config.MODEL_CONFIGS`
- Evaluate on the dev set after each epoch
- Save the best model for each configuration
- Log training progress and results

## Model Configurations

The following models are implemented:

1. **Dot Product**: Simple dot product similarity (no trainable parameters)
2. **Weighted Dot Product**: Element-wise weighted dot product
3. **Low-Rank Bilinear**: Factorized bilinear form with ranks 32, 64, 128
4. **Full-Rank Bilinear**: Full bilinear form (memory intensive)

## Results and Analysis

After training, results are saved in:
- `saved_models/{model_name}/eval_results.txt`: Text summary
- `saved_models/{model_name}/results.json`: JSON format for programmatic access
- `saved_models/{model_name}/training.log`: Detailed training logs
- `saved_models/summary_results.json`: Overall summary of all models

## Customization

### Adding New Models

1. Define the model class in `models.py`
2. Add the model configuration to `config.MODEL_CONFIGS`
3. Update the `get_model()` function if needed

### Hyperparameter Tuning

Edit `config.py` to modify:
- Learning rate, batch size, number of epochs
- Model-specific parameters (ranks for low-rank models)
- Training parameters (margin for loss function)

### Different SBERT Models

To use a different SBERT model:
1. Update `SBERT_MODEL_NAME` in `config.py`
2. Update `EMBEDDING_DIM` to match the new model
3. Regenerate embeddings with the new model

## Troubleshooting

### Common Issues

1. **FileNotFoundError during embedding loading**
   - Ensure you've run `preprocess_embeddings.py` successfully
   - Check that all files exist in the `embeddings/` directory

2. **CUDA out of memory**
   - Reduce `BATCH_SIZE` in `config.py`
   - Use a smaller SBERT model for preprocessing
   - Consider using CPU for preprocessing (slower but uses less memory)

3. **Evaluation script errors**
   - Ensure the MS MARCO evaluation script is downloaded
   - Check that the path in `MSMARCO_EVAL_SCRIPT` is correct
   - Verify the qrels file exists and has the correct format

4. **Slow preprocessing**
   - Preprocessing is CPU/GPU intensive and takes time
   - Consider using a smaller subset for initial experiments
   - Enable GPU if available (`DEVICE="cuda"` in config)

### Debugging

Use the verification script to check embeddings:
```bash
python preprocess_embeddings.py --verify-only
```

Check training with a small dataset:
```python
# In main_train.py, set:
train_dataset_limit = 1000
```

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
sentence-transformers>=2.2.0
tqdm>=4.62.0
```

## Citation

If you use this code, please cite the relevant papers for MS MARCO and sentence-transformers.