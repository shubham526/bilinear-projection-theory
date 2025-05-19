# experiment1/config.py

import torch

# --- Experiment Parameters ---
DIM_N = 10
NUM_TEST_CASES_BILINEAR = 1000
NUM_TEST_CASES_WDP_GENERALIZATION = 1000

# --- WDP Training Base Hyperparameters ---
# These will be used unless overridden by a specific run configuration
BASE_WDP_LEARNING_RATE = 1e-3
BASE_WDP_MARGIN = 1.0
BASE_WDP_BATCH_SIZE = 128

# --- Run Configurations for Experiment 1 ---
# Define different scenarios you want to run.
# Each scenario can have its own WDP training settings.
RUN_CONFIGS = {
    "default_wdp_training": {
        "train_wdp_flag": True,
        "wdp_epochs": 5,
        "wdp_train_samples": 50000,
        "wdp_learning_rate": BASE_WDP_LEARNING_RATE, # Use base or override
        "wdp_batch_size": BASE_WDP_BATCH_SIZE,       # Use base or override
        "wdp_margin": BASE_WDP_MARGIN,               # Use base or override
        "description": "Default WDP training settings."
    },
    "more_epochs_wdp": {
        "train_wdp_flag": True,
        "wdp_epochs": 15, # Increased epochs
        "wdp_train_samples": 50000,
        "wdp_learning_rate": BASE_WDP_LEARNING_RATE,
        "wdp_batch_size": BASE_WDP_BATCH_SIZE,
        "wdp_margin": BASE_WDP_MARGIN,
        "description": "WDP training with more epochs."
    },
    "more_samples_wdp": {
        "train_wdp_flag": True,
        "wdp_epochs": 5, # Back to default epochs, or keep higher if combining
        "wdp_train_samples": 100000, # Increased samples
        "wdp_learning_rate": BASE_WDP_LEARNING_RATE,
        "wdp_batch_size": BASE_WDP_BATCH_SIZE,
        "wdp_margin": BASE_WDP_MARGIN,
        "description": "WDP training with more samples."
    },
    "default_univar_wdp_no_train": { # For testing the WDP generalization part without training it first
        "train_wdp_flag": False, # This will make train_single_wdp_model skip training for the main WDP
        "wdp_epochs": 0, # Not applicable
        "wdp_train_samples": 0, # Not applicable
        "description": "Uses a default (untrained, weights=1) WDP for generalization tests."
    }
}

# List of random seeds to apply to each configuration in RUN_CONFIGS
# For each config in RUN_CONFIGS, the experiment will be run once per seed in this list.
SEEDS_TO_RUN = [42, 123, 789] # Add more seeds if desired

# --- Other Parameters ---
SAVE_DETAILED_RESULTS = True
ANALYZE_FAILURE_PATTERNS = True
MIN_SAMPLES_FOR_I0_ANALYSIS = 5
CONFIDENCE_LEVEL = 0.95
NUM_SAMPLES_PER_I0_TEST = 200
INCLUDE_SPECIALIZED_WDP_TEST = True # This trains its own WDP models internally

# +++ Parameters for Specialized WDP Training (within test_wdp_with_multiple_fixed_I0_sets) +++
SPECIALIZED_WDP_EPOCHS = 3  # Fewer epochs for specialized models as they have a simpler target
SPECIALIZED_WDP_TRAIN_SAMPLES_FACTOR = 0.2 # Factor of general WDP_TRAIN_SAMPLES (e.g., 50000*0.2 = 10000)
# +++ END OF ADDED/MODIFIED PARAMETERS FOR SPECIALIZED WDP +++


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR_EXP1 = "/home/shubham-chatterjee/PycharmProjects/bilinear-proj-theory/results/experiment1/saved_models_exp1/" # Make sure this path is correct

# +++ ADDED: Global variable to hold current run's WDP training params +++
# This is a simple way to pass them to train_single_wdp_model without changing its signature too much
# or passing config dicts around everywhere.
CURRENT_WDP_TRAIN_PARAMS = {}