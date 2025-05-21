# utils.py
import os
import json
import logging
import time

def load_folds(folds_file_path):
    """
    Load cross-validation folds from JSON file.

    Args:
        folds_file_path: Path to folds JSON file

    Returns:
        List of fold dictionaries with 'training' and 'testing' keys
    """
    if not os.path.exists(folds_file_path):
        raise FileNotFoundError(f"Folds file not found: {folds_file_path}")

    with open(folds_file_path, 'r') as f:
        folds = json.load(f)

    return folds


def setup_logging(model_save_dir):
    """Setup logging for training"""
    os.makedirs(model_save_dir, exist_ok=True)

    # Create a unique logger for this model
    logger_name = f"model_{os.path.basename(model_save_dir)}_{int(time.time())}"
    logger = logging.getLogger(logger_name)

    # Important: Set propagate to False to avoid duplicate logs
    logger.propagate = False

    # Reset handlers if any exist already
    if logger.handlers:
        logger.handlers = []

    # Set logging level
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(model_save_dir, 'training.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger