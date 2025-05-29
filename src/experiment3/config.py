# experiment3/config.py
import torch
import os

# --- Experiment 3: Low-Rank Approximation Analysis ---
EXP3_ENABLED = True  # Set to False to skip this experiment if running a master script

# --- Dataset Configuration ---
# Specify which dataset this experiment should analyze.
# Options: "msmarco", "car", "robust"
# If not specified, will try to infer from model path
DATASET_NAME = "msmarco"  # Change this to "car" or "robust" if analyzing those datasets
# --- Path to the TRAINED Full-Rank Bilinear Model (or a high-rank LRB to approximate W*) ---
# This model should have been saved by experiment2 training
# The paths will be searched in order, first match wins

# For MS MARCO models
MSMARCO_MODEL_PATHS = [
    "/home/user/sisap2025/results/experiment2/facebook-contriever/msmarco/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/results/experiment2/bert-base-uncased/msmarco/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/results/experiment2/msmarco/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/experiment2/saved_models/msmarco/full_rank_bilinear/best_model.pth",
    "/home/user/bilinear-projection-theory/results/experiment2/msmarco/full_rank_bilinear/best_model.pth"
]

# For CAR models
CAR_MODEL_PATHS = [
    "/home/user/sisap2025/results/experiment2/facebook-contriever/car/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/results/experiment2/bert-base-uncased/car/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/results/experiment2/car/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/experiment2/saved_models/car/full_rank_bilinear/best_model.pth",
]

# For ROBUST models
ROBUST_MODEL_PATHS = [
    "/home/user/sisap2025/results/experiment2/facebook-contriever/robust/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/results/experiment2/bert-base-uncased/robust/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/results/experiment2/robust/full_rank_bilinear/best_model.pth",
    "/home/user/sisap2025/experiment2/saved_models/robust/full_rank_bilinear/best_model.pth",
]

# Combine all possible paths based on dataset
ALL_POSSIBLE_PATHS = MSMARCO_MODEL_PATHS + CAR_MODEL_PATHS + ROBUST_MODEL_PATHS

# Find the first existing model path
PRETRAINED_W_STAR_MODEL_PATH = None
dataset_specific_paths = []

if DATASET_NAME == "msmarco":
    dataset_specific_paths = MSMARCO_MODEL_PATHS
elif DATASET_NAME == "car":
    dataset_specific_paths = CAR_MODEL_PATHS
elif DATASET_NAME == "robust":
    dataset_specific_paths = ROBUST_MODEL_PATHS

# First try dataset-specific paths, then all paths
search_paths = dataset_specific_paths + [p for p in ALL_POSSIBLE_PATHS if p not in dataset_specific_paths]

for path in search_paths:
    if os.path.exists(path):
        PRETRAINED_W_STAR_MODEL_PATH = path
        break

# If no model found, set to the most likely location for the specified dataset
if PRETRAINED_W_STAR_MODEL_PATH is None:
    if DATASET_NAME == "msmarco":
        PRETRAINED_W_STAR_MODEL_PATH = "/home/user/sisap2025/results/experiment2/msmarco/full_rank_bilinear/best_model.pth"
    elif DATASET_NAME == "car":
        PRETRAINED_W_STAR_MODEL_PATH = "/home/user/sisap2025/results/experiment2/car/full_rank_bilinear/best_model.pth"
    elif DATASET_NAME == "robust":
        PRETRAINED_W_STAR_MODEL_PATH = "/home/user/sisap2025/results/experiment2/robust/full_rank_bilinear/best_model.pth"
    else:
        PRETRAINED_W_STAR_MODEL_PATH = "/home/user/sisap2025/results/experiment2/full_rank_bilinear/best_model.pth"

    print(f"Warning: No trained model found for {DATASET_NAME}. Please update paths in experiment3/config.py")
    print(f"Current path set to: {PRETRAINED_W_STAR_MODEL_PATH}")
    print(f"Searched locations: {search_paths}")

# Specify the model key from experiment2/config.py's MODEL_CONFIGS
# that PRETRAINED_W_STAR_MODEL_PATH corresponds to.
PRETRAINED_W_STAR_MODEL_KEY = "full_rank_bilinear"

# --- Parameters for W_r Approximations ---
# Ranks 'r' to test for W_r approximations.
# Ensure ranks do not exceed the original embedding dimension or rank of W_star.
EXP3_RANKS_TO_TEST = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

# --- Output Directory ---
EXP3_RESULTS_DIR = f"/home/user/sisap2025/results/experiment3/{DATASET_NAME}_analysis/"

# --- Pointwise Error Verification (Optional) ---
VERIFY_POINTWISE_ERROR_BOUND = True
NUM_POINTWISE_ERROR_SAMPLES = 10000  # Reduced from 100000 for faster execution

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset-specific settings ---
# These help experiment3 understand how to load and evaluate data for each dataset

DATASET_CONFIGS = {
    "msmarco": {
        "use_ir_datasets": True,
        "primary_metric": "mrr_cut_10",  # MRR@10 for MS MARCO
        "dataset_name_for_eval": "msmarco-passage"
    },
    "car": {
        "use_ir_datasets": False,  # Uses file-based loading
        "primary_metric": "map",  # MAP for TREC CAR
        "dataset_name_for_eval": "car"
    },
    "robust": {
        "use_ir_datasets": False,  # Uses file-based loading
        "primary_metric": "ndcg_cut_10",  # nDCG@10 for TREC ROBUST
        "dataset_name_for_eval": "robust"
    }
}

# Get config for current dataset
CURRENT_DATASET_CONFIG = DATASET_CONFIGS.get(DATASET_NAME, DATASET_CONFIGS["msmarco"])

print(f"Experiment 3 config loaded successfully.")
print(f"Dataset: {DATASET_NAME}")
print(f"Model path: {PRETRAINED_W_STAR_MODEL_PATH}")
print(f"Model exists: {os.path.exists(PRETRAINED_W_STAR_MODEL_PATH) if PRETRAINED_W_STAR_MODEL_PATH else False}")
print(f"Results directory: {EXP3_RESULTS_DIR}")
print(f"Primary metric: {CURRENT_DATASET_CONFIG['primary_metric']}")
print(f"Use ir_datasets: {CURRENT_DATASET_CONFIG['use_ir_datasets']}")
print(f"Device: {DEVICE}")

# Create results directory
os.makedirs(EXP3_RESULTS_DIR, exist_ok=True)