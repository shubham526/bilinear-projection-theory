# config.py

import torch

# --- Data Paths ---
# Student needs to update these paths
EMBEDDING_DIR = "/home/user/embeddings/sbert_all-mpnet-base-v2/"  # Parent directory for fixed embeddings

QUERY_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}query_embeddings.npy"  # NumPy array
PASSAGE_EMBEDDINGS_PATH = f"{EMBEDDING_DIR}passage_embeddings.npy"  # NumPy array
QUERY_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}query_id_to_idx.json"  # Maps qid to embedding row index
PASSAGE_ID_TO_IDX_PATH = f"{EMBEDDING_DIR}passage_id_to_idx.json"  # Maps pid to embedding row index

# --- Model Configs ---
# Assuming SBERT 'all-mpnet-base-v2' which has 768 dimensions
EMBEDDING_DIM = 768

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
BATCH_SIZE = 64  # For query-positive-negative triples
NUM_EPOCHS = 3
WEIGHT_DECAY = 1e-5
MARGIN = 1.0  # For MarginRankingLoss
EVAL_BATCH_SIZE = 256  # For scoring candidates during evaluation
MODEL_SAVE_DIR = "experiment2/saved_models/"
LOG_INTERVAL = 100  # Log training loss every N batches

# --- Evaluation ---
# Add this if you want to store additional metrics
METRICS_TO_EVALUATE = [
    'mrr_cut.100', 'mrr_cut.1000',
    'recip_rank',
    'recall.100', 'recall.1000',
    'ndcg_cut.10', 'ndcg_cut.100'
]

# --- SBERT Model for preprocessing ---
SBERT_MODEL_NAME = 'all-mpnet-base-v2'  # The SBERT model to use for embedding generation