# config.py
import torch

# --- Experiment Parameters ---
DIM_N = 10  # Dimension of the hypercube vectors (n >= 3 for WDP failure demonstration)
NUM_TEST_CASES_BILINEAR = 1000 # Number of random (q, I0) pairs to test for bilinear model
NUM_TEST_CASES_WDP_GENERALIZATION = 1000 # Number of (q, I0) pairs to test a single trained WDP on

# --- Weighted Dot Product (WDP) Training (if attempting to train one) ---
# These are only used if you choose to train a single WDP model to test its universality
WDP_TRAIN_SAMPLES = 50000     # Number of (q, I0, d_agree, d_disagree) samples for training WDP
WDP_LEARNING_RATE = 1e-3
WDP_EPOCHS = 5
WDP_BATCH_SIZE = 128
WDP_MARGIN = 1.0 # Margin for MarginRankingLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_DIR_EXP1 = "saved_models_exp1/"
