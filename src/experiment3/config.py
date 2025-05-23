# experiment3/config.py
import torch

# --- Experiment 3: Low-Rank Approximation Analysis ---
EXP3_ENABLED = True  # Set to False to skip this experiment if running a master script

# --- Path to the TRAINED Full-Rank Bilinear Model (or a high-rank LRB to approximate W*) ---
# This model should have been saved by experiment2/main_train.py
# STUDENT: CRITICALLY UPDATE THIS PATH AND KEY
# Example: "experiment2/saved_models/full_rank_bilinear/best_model.pth"
# Example if using an LRB as W_star: "experiment2/saved_models/low_rank_bilinear_128/best_model.pth"
PRETRAINED_W_STAR_MODEL_PATH = "/home/user/sisap2025/results/experiment2/bert-base-uncased/msmarco/full_rank_bilinear/best_model.pth"

# Specify the model key from experiment2/config.py's MODEL_CONFIGS
# that PRETRAINED_W_STAR_MODEL_PATH corresponds to.
# This helps in correctly loading the model architecture before loading state_dict.
# Example: "full_rank_bilinear" or "low_rank_bilinear_128"
PRETRAINED_W_STAR_MODEL_KEY = "full_rank_bilinear"

# --- Parameters for W_r Approximations ---
# Ranks 'r' to test for W_r approximations.
# Example: list(range(1, 33, 2)) + [32, 48, 64, 96, 128]
# Ensure ranks do not exceed the original embedding dimension or rank of W_star.
EXP3_RANKS_TO_TEST = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

# --- Output Directory ---
EXP3_RESULTS_DIR = "/home/user/sisap2025/results/experiment3/saved_results_exp3/"

# --- Pointwise Error Verification (Optional) ---
VERIFY_POINTWISE_ERROR_BOUND = True
NUM_POINTWISE_ERROR_SAMPLES = 100000 # Number of (q,d) pairs to sample for verification

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Paths to Experiment 2 Data (for evaluation) ---
# These paths point to where Experiment 2 stores its data and embeddings.
# Assumes this script is run from the project root (bilinear-proj-theory/)
# So, paths to experiment2 files are relative to that.
# If experiment2/config.py is imported, these might not be needed here if that config is used directly.
# For clarity and explicitness for Exp3, we can redefine or ensure they are accessible.

# Path to the Experiment 2 configuration file to load its settings
# This helps avoid duplicating all data paths.
# IMPORTANT: Ensure this path is correct relative to where main_experiment3.py is run from.
# If running `python -m experiment3.main_experiment3` from project root, this should work.
PATH_TO_EXP2_CONFIG_DIR = "/home/shubham-chatterjee/PycharmProjects/bilinear-proj-theory/src/experiment2"

# The following will be loaded dynamically using experiment2's config
# DEV_QUERIES_PATH (from exp2_config)
# DEV_CANDIDATES_PATH (from exp2_config)
# DEV_QRELS_PATH (from exp2_config)
# QUERY_EMBEDDINGS_PATH (from exp2_config)
# PASSAGE_EMBEDDINGS_PATH (from exp2_config)
# QUERY_ID_TO_IDX_PATH (from exp2_config)
# PASSAGE_ID_TO_IDX_PATH (from exp2_config)
# MSMARCO_EVAL_SCRIPT (from exp2_config)
# EMBEDDING_DIM (from exp2_config)