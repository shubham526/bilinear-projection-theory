# config.py

import torch

# --- Data Paths ---
# Student needs to update these paths
MSMARCO_V1_DIR = "data/msmarco_v1/"  # Parent directory for MS MARCO files
EMBEDDING_DIR = "embeddings/sbert_all-mpnet-base-v2/"  # Parent directory for fixed embeddings

TRAIN_TRIPLES_PATH = f"{MSMARCO_V1_DIR}triples.train.small.tsv"  # Or a subset file they create
DEV_QUERIES_PATH = f"{MSMARCO_V1_DIR}queries.dev.small.tsv"  # Standard MS MARCO dev queries
DEV_QRELS_PATH = f"{MSMARCO_V1_DIR}qrels.dev.tsv"
DEV_CANDIDATES_PATH = f"{MSMARCO_V1_DIR}top1000.dev.txt"  # Standard candidate file
CORPUS_PATH = f"{MSMARCO_V1_DIR}collection.tsv"  # Full passage collection

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
    # "full_rank_bilinear": {
    #     "type": "full_rank_bilinear",
    #     "embedding_dim": EMBEDDING_DIM
    # }
}

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64  # For query-positive-negative triples
NUM_EPOCHS = 3
WEIGHT_DECAY = 1e-5
MARGIN = 1.0  # For MarginRankingLoss
EVAL_BATCH_SIZE = 256  # For scoring candidates during evaluation
MODEL_SAVE_DIR = "saved_models/"
LOG_INTERVAL = 100  # Log training loss every N batches

# --- Evaluation ---
# Path to the official MS MARCO evaluation script
MSMARCO_EVAL_SCRIPT = "ms_marco_eval/msmarco_passage_eval.py"

# --- SBERT Model for preprocessing ---
SBERT_MODEL_NAME = 'all-mpnet-base-v2'  # The SBERT model to use for embedding generation