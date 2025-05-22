# config.py

import os
import torch

# --- Dataset Settings ---
DATASET_NAME = "msmarco"  # Default dataset if not specified

# --- Data Paths ---
# Student needs to update these paths
EMBEDDING_DIR = "/home/user/sisap2025/embeddings/bert-base-uncased/msmarco/"  # Parent directory for fixed embeddings

# Standard MS MARCO paths
QUERY_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}query_embeddings.npy"  # NumPy array
PASSAGE_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}passage_embeddings.npy"  # NumPy array
QUERY_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}query_id_to_idx.json"  # Maps qid to embedding row index
PASSAGE_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}passage_id_to_idx.json"  # Maps pid to embedding row index

# TREC CAR paths
CAR_QUERY_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}car_query_embeddings.npy"
CAR_PASSAGE_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}car_passage_embeddings.npy"
CAR_QUERY_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}car_query_id_to_idx.json"
CAR_PASSAGE_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}car_passage_id_to_idx.json"

# CAR data files
CAR_QUERIES_FILE = "/home/user/sisap2025/data/car/queries.tsv"  # Update this path
CAR_QRELS_FILE = "/home/user/sisap2025/data/car/qrels.txt"  # Update this path
CAR_RUN_FILE = "/home/user/sisap2025/data/car/run.txt"  # Update this path
CAR_FOLDS_FILE = "/home/user/sisap2025/data/car/folds.json"  # Update this path

# TREC ROBUST paths
ROBUST_QUERY_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}robust_query_embeddings.npy"
ROBUST_PASSAGE_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}robust_passage_embeddings.npy"
ROBUST_QUERY_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}robust_query_id_to_idx.json"
ROBUST_PASSAGE_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}robust_passage_id_to_idx.json"

# ROBUST data files
ROBUST_QUERIES_FILE = "/home/user/sisap2025/data/robust/queries.tsv"  # Update this path
ROBUST_QRELS_FILE = "/home/user/sisap2025/data/robust/qrels.txt"  # Update this path
ROBUST_RUN_FILE = "/home/user/sisap2025/data/robust/run.txt"  # Update this path
ROBUST_FOLDS_FILE = "/home/user/sisap2025/data/robust/folds.json"  # Update this path

ROBUST_USE_CHUNKING = True
ROBUST_CHUNK_SIZE = 512
ROBUST_CHUNK_STRIDE = 256
ROBUST_CHUNK_AGGREGATION = "hybrid"

# --- Model for preprocessing ---
EMBEDDING_MODEL_NAME = 'bert-base-uncased'  # or any HuggingFace model name
EMBEDDING_DIM = 768
# SBERT_MODEL_NAME = 'all-mpnet-base-v2'  # The SBERT model to use for embedding generation



# --- Model Configs ---

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
    # Uncomment if you want to experiment with full rank (memory intensive)
    "full_rank_bilinear": {
        "type": "full_rank_bilinear",
        "embedding_dim": EMBEDDING_DIM
    }
}

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 256  # For query-positive-negative triples
NUM_EPOCHS = 8
WEIGHT_DECAY = 1e-5
MARGIN = 1.0  # For MarginRankingLoss
EVAL_BATCH_SIZE = 256  # For scoring candidates during evaluation
MODEL_SAVE_DIR = "/home/user/sisap2025/experiment2/saved_models/"
LOG_INTERVAL = 100  # Log training loss every N batches

# --- Cross-validation Training Settings ---
CV_BATCH_SIZE = 256
CV_NUM_NEGATIVES = 3  # Number of negative samples per positive

# --- Evaluation ---
# Add this if you want to store additional metrics
METRICS_TO_EVALUATE = [
    'ndcg_cut_10', 'ndcg_cut_100',
    'recall_100', 'recall_1000',
    'map',
    'recip_rank'
]
