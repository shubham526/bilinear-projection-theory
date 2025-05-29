# experiment3/experiment3.py - Fixed version with all bugs resolved

import torch
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
import pandas as pd
import gc
from contextlib import contextmanager

# Get the absolute path to project root (two levels up from this file)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

# Add paths to sys.path
paths_to_add = [
    project_root,
    os.path.join(project_root, 'src'),
    os.path.join(project_root, 'src', 'experiment1'),
    os.path.join(project_root, 'src', 'experiment2'),
    os.path.join(project_root, 'src', 'experiment3'),
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Print debug info to verify paths
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path entries added: {len(paths_to_add)}")

# Import local config explicitly from experiment3 directory
config_path = os.path.join(current_file_dir, 'config.py')

print(f"Looking for config at: {config_path}")
print(f"Config file exists: {os.path.exists(config_path)}")

if os.path.exists(config_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp3_config", config_path)
    exp3_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp3_config)
    print(f"✓ Loaded config. EXP3_ENABLED = {getattr(exp3_config, 'EXP3_ENABLED', 'NOT FOUND')}")
else:
    print("✗ Config file not found, using fallback")


    # Fallback config
    class FallbackConfig:
        EXP3_ENABLED = True
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EXP3_RESULTS_DIR = "results/experiment3"
        PRETRAINED_W_STAR_MODEL_PATH = "results/experiment2/models/full_rank_bilinear_best.pth"
        PRETRAINED_W_STAR_MODEL_KEY = "full_rank_bilinear"
        EXP3_RANKS_TO_TEST = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        VERIFY_POINTWISE_ERROR_BOUND = True
        NUM_POINTWISE_ERROR_SAMPLES = 1000
        DATASET_NAME = "msmarco"


    exp3_config = FallbackConfig()


# Fixed import strategy - try multiple approaches
def import_dependencies():
    """Import all required dependencies with proper fallback"""
    global get_exp2_model, FullRankBilinearModel, LowRankBilinearModel
    global load_embeddings_and_mappings, load_dev_data_for_eval
    global evaluate_model_on_dev, exp2_config, BilinearScorer

    # Strategy 1: Direct imports (works if modules are properly installed)
    try:
        from models import get_model as get_exp2_model, FullRankBilinearModel, LowRankBilinearModel
        from data_loader import load_embeddings_and_mappings, load_dev_data_for_eval
        from evaluate import evaluate_model_on_dev
        import config as exp2_config

        # Try experiment1 import
        try:
            from experiment1.models import BilinearScorer
        except ImportError:
            # Fallback: import from models if experiment1 package import fails
            import importlib.util
            exp1_models_path = os.path.join(project_root, 'src', 'experiment1', 'models.py')
            if os.path.exists(exp1_models_path):
                spec = importlib.util.spec_from_file_location("exp1_models", exp1_models_path)
                exp1_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(exp1_models)
                BilinearScorer = exp1_models.BilinearScorer
            else:
                raise ImportError("Cannot find BilinearScorer")

        print("✓ All imports successful!")
        return True

    except ImportError as e:
        print(f"✗ Direct import failed: {e}")
        print("Trying fallback import strategy...")

    # Strategy 2: Module-specific imports with explicit paths
    try:
        import importlib.util

        # Import experiment2 models
        exp2_models_path = os.path.join(project_root, 'src', 'experiment2', 'models.py')
        if not os.path.exists(exp2_models_path):
            raise ImportError(f"Cannot find experiment2 models at {exp2_models_path}")

        spec = importlib.util.spec_from_file_location("exp2_models", exp2_models_path)
        exp2_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp2_models)

        get_exp2_model = exp2_models.get_model
        FullRankBilinearModel = exp2_models.FullRankBilinearModel
        LowRankBilinearModel = exp2_models.LowRankBilinearModel

        # Import experiment2 data_loader
        exp2_data_loader_path = os.path.join(project_root, 'src', 'experiment2', 'data_loader.py')
        if not os.path.exists(exp2_data_loader_path):
            raise ImportError(f"Cannot find experiment2 data_loader at {exp2_data_loader_path}")

        spec = importlib.util.spec_from_file_location("exp2_data_loader", exp2_data_loader_path)
        exp2_data_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp2_data_loader)

        load_embeddings_and_mappings = exp2_data_loader.load_embeddings_and_mappings
        load_dev_data_for_eval = exp2_data_loader.load_dev_data_for_eval

        # Import experiment2 evaluate
        exp2_evaluate_path = os.path.join(project_root, 'src', 'experiment2', 'evaluate.py')
        if not os.path.exists(exp2_evaluate_path):
            raise ImportError(f"Cannot find experiment2 evaluate at {exp2_evaluate_path}")

        spec = importlib.util.spec_from_file_location("exp2_evaluate", exp2_evaluate_path)
        exp2_evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp2_evaluate)

        evaluate_model_on_dev = exp2_evaluate.evaluate_model_on_dev

        # Import experiment2 config
        exp2_config_path = os.path.join(project_root, 'src', 'experiment2', 'config.py')
        if not os.path.exists(exp2_config_path):
            raise ImportError(f"Cannot find experiment2 config at {exp2_config_path}")

        spec = importlib.util.spec_from_file_location("exp2_config_module", exp2_config_path)
        exp2_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp2_config)

        # Import experiment1 models
        exp1_models_path = os.path.join(project_root, 'src', 'experiment1', 'models.py')
        if not os.path.exists(exp1_models_path):
            raise ImportError(f"Cannot find experiment1 models at {exp1_models_path}")

        spec = importlib.util.spec_from_file_location("exp1_models", exp1_models_path)
        exp1_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp1_models)
        BilinearScorer = exp1_models.BilinearScorer

        print("✓ Fallback imports successful!")
        return True

    except Exception as fallback_error:
        print(f"✗ Fallback import also failed: {fallback_error}")
        raise ImportError(f"Could not import required dependencies: {fallback_error}")


# Import all dependencies
import_dependencies()


@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def determine_dataset_info(model_path):
    """
    Fixed: Determine the dataset name based on config and model path.
    Returns (dataset_name, use_ir_datasets)
    """
    logger = logging.getLogger('experiment3')

    # Start with config if available, but allow path to override
    config_dataset = getattr(exp3_config, 'DATASET_NAME', None)

    # Try to infer from model path
    inferred_dataset = None
    if model_path:
        model_path_lower = model_path.lower()
        if 'msmarco' in model_path_lower:
            inferred_dataset = "msmarco"
        elif 'car' in model_path_lower:
            inferred_dataset = "car"
        elif 'robust' in model_path_lower:
            inferred_dataset = "robust"

    # Decision logic: use inferred if available, otherwise use config, otherwise default
    if inferred_dataset:
        dataset_name = inferred_dataset
        logger.info(f"Using dataset inferred from model path: {dataset_name}")
    elif config_dataset:
        dataset_name = config_dataset
        logger.info(f"Using dataset from config: {dataset_name}")
    else:
        dataset_name = "msmarco"
        logger.warning(f"Could not determine dataset from path or config, defaulting to 'msmarco'")

    # Normalize dataset name and determine data loading strategy
    if dataset_name in ["msmarco", "msmarco-passage"]:
        use_ir_datasets = True
        dataset_name = "msmarco"  # Keep it simple
    else:
        use_ir_datasets = False

    logger.info(f"Final dataset: {dataset_name}, use_ir_datasets: {use_ir_datasets}")
    return dataset_name, use_ir_datasets


def setup_logging_exp3(results_dir):
    """Fixed: Setup module-specific logging for Experiment 3"""
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, 'experiment3.log')

    # Create a specific logger for experiment3 (don't modify root logger)
    logger = logging.getLogger('experiment3')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_W_star_matrix(model_path, model_key_from_exp2_config, exp2_model_configs):
    """
    Fixed: Loads a pre-trained model and extracts its W* matrix with proper validation.
    """
    logger = logging.getLogger('experiment3')
    logger.info(f"Loading W* from model path: {model_path} (type: {model_key_from_exp2_config})")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Validate that file is a valid PyTorch checkpoint
    try:
        checkpoint_test = torch.load(model_path, map_location='cpu')
        if not isinstance(checkpoint_test, dict):
            raise ValueError("Invalid checkpoint format")
        logger.info("Model file validation passed")
    except Exception as e:
        logger.error(f"Invalid model file: {e}")
        raise ValueError(f"Invalid model file: {e}")

    model_config_params = exp2_model_configs.get(model_key_from_exp2_config)
    if not model_config_params:
        logger.error(f"Model key '{model_key_from_exp2_config}' not found in Experiment 2 MODEL_CONFIGS.")
        available_keys = list(exp2_model_configs.keys())
        logger.error(f"Available model keys: {available_keys}")
        raise ValueError(f"Model key '{model_key_from_exp2_config}' not found. Available: {available_keys}")

    # Instantiate model structure
    try:
        model_architecture = get_exp2_model(model_key_from_exp2_config, model_config_params)
    except Exception as e:
        logger.error(f"Failed to create model architecture: {e}")
        raise

    try:
        checkpoint = torch.load(model_path, map_location=exp3_config.DEVICE)
        if 'model_state_dict' in checkpoint:
            model_architecture.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model_state_dict from checkpoint.")
        else:
            model_architecture.load_state_dict(checkpoint)
            logger.info("Loaded state_dict directly.")
    except Exception as e:
        logger.error(f"Error loading model state_dict: {e}")
        raise

    model_architecture.eval().to(exp3_config.DEVICE)

    W_star = None
    if isinstance(model_architecture, FullRankBilinearModel):
        W_star = model_architecture.W.data.clone()
        logger.info(f"Extracted W* directly from FullRankBilinearModel. Shape: {W_star.shape}")
    elif isinstance(model_architecture, LowRankBilinearModel):
        P = model_architecture.P.data
        Q = model_architecture.Q.data
        W_star = torch.matmul(P, Q.T).clone()
        logger.info(f"Reconstructed W* = P@Q.T from LowRankBilinearModel. Shape: {W_star.shape}")
        logger.info(f"  Original rank of LRB model was: {model_architecture.rank}")
    else:
        msg = (f"Unsupported model type '{type(model_architecture)}' for W* extraction. "
               "Expected FullRankBilinearModel or LowRankBilinearModel.")
        logger.error(msg)
        raise TypeError(msg)

    return W_star.cpu()  # Move to CPU for SVD


def perform_svd(W_star_matrix_cpu):
    """Performs SVD on W* and returns U, S, Vh."""
    logger = logging.getLogger('experiment3')
    logger.info("Performing SVD on W* matrix (on CPU)...")

    try:
        U, S_singular_values, Vh = torch.linalg.svd(W_star_matrix_cpu.float())
        logger.info(f"SVD completed. U shape: {U.shape}, S shape: {S_singular_values.shape}, Vh shape: {Vh.shape}")
        return U, S_singular_values, Vh
    except Exception as e:
        logger.error(f"SVD failed: {e}")
        raise


def construct_Wr_approximation(U, S_singular_values, Vh, r):
    """Fixed: Constructs rank-r approximation W_r = U_r @ diag(S_r) @ Vh_r."""
    logger = logging.getLogger('experiment3')

    if r < 1:
        raise ValueError("Rank r must be at least 1.")

    max_rank = min(S_singular_values.shape[0], U.shape[1], Vh.shape[0])
    actual_r = min(r, max_rank)

    if actual_r != r:
        logger.warning(f"Requested rank r={r} adjusted to actual_r={actual_r} due to matrix dimensions.")

    U_r = U[:, :actual_r]
    S_r_diag = torch.diag(S_singular_values[:actual_r])
    Vh_r = Vh[:actual_r, :]
    W_r = torch.matmul(U_r, torch.matmul(S_r_diag, Vh_r))

    return W_r


def plot_singular_values(singular_values, save_path_dir, filename="singular_values_spectrum.png"):
    """Fixed: Plot singular values with better error handling"""
    logger = logging.getLogger('experiment3')
    logger.info("Plotting singular value spectrum...")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(singular_values) + 1), singular_values.numpy(), marker='o', linestyle='-')
        plt.yscale('log')
        plt.title('Singular Value Spectrum of W*')
        plt.xlabel('Singular Value Index (i)')
        plt.ylabel('Singular Value (σ_i) - Log Scale')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()

        plot_file = os.path.join(save_path_dir, filename)
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Singular value plot saved to {plot_file}")
    except Exception as e:
        logger.error(f"Failed to create singular values plot: {e}")


def plot_performance_vs_rank(ranks_tested, scores, save_path_dir, metric_name="MRR@10",
                             filename="performance_vs_rank.png"):
    """Fixed: Plot performance vs rank with better formatting"""
    logger = logging.getLogger('experiment3')
    logger.info(f"Plotting {metric_name} vs. Rank r...")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(ranks_tested, scores, marker='x', linestyle='-', linewidth=2, markersize=8)
        plt.title(f'Retrieval Performance ({metric_name}) vs. Rank r of W_r')
        plt.xlabel('Rank (r)')
        plt.ylabel(f'{metric_name}')

        # Fixed: Better x-tick handling
        if len(ranks_tested) > 10:
            step = max(1, len(ranks_tested) // 8)
            selected_ticks = ranks_tested[::step]
            if ranks_tested[-1] not in selected_ticks:
                selected_ticks.append(ranks_tested[-1])
            plt.xticks(selected_ticks)
        else:
            plt.xticks(ranks_tested)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = os.path.join(save_path_dir, filename)
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance vs. Rank plot saved to {plot_file}")
    except Exception as e:
        logger.error(f"Failed to create performance vs rank plot: {e}")


def plot_performance_vs_sigma_r_plus_1(ranks_tested, scores, singular_values, save_path_dir,
                                       metric_name="MRR@10", filename="performance_vs_sigma_r_plus_1.png"):
    """Fixed: Plot performance vs sigma with better data validation"""
    logger = logging.getLogger('experiment3')
    logger.info(f"Plotting {metric_name} vs. σ_{{r+1}}...")

    sigma_r_plus_1_values = []
    valid_scores_for_sigma_plot = []

    for i, r_val in enumerate(ranks_tested):
        if r_val < len(singular_values):
            sigma_r_plus_1_values.append(singular_values[r_val].item())
            valid_scores_for_sigma_plot.append(scores[i])

    if len(sigma_r_plus_1_values) < 2:
        logger.warning(f"Not enough data to plot performance vs sigma_r+1 (only {len(sigma_r_plus_1_values)} points).")
        return

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(sigma_r_plus_1_values, valid_scores_for_sigma_plot, marker='o', linestyle='-', linewidth=2,
                 markersize=8)
        plt.title(f'Retrieval Performance ({metric_name}) vs. Next Neglected Singular Value (σ_{{r+1}})')
        plt.xlabel('Next Singular Value (σ_{r+1}) - Log Scale')
        plt.ylabel(f'{metric_name}')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.3)

        # Invert x-axis for better visualization (larger sigma values on left)
        if len(sigma_r_plus_1_values) > 1 and sigma_r_plus_1_values[0] > sigma_r_plus_1_values[-1]:
            plt.gca().invert_xaxis()

        plt.tight_layout()
        plot_file = os.path.join(save_path_dir, filename)
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance vs. Sigma_r+1 plot saved to {plot_file}")
    except Exception as e:
        logger.error(f"Failed to create performance vs sigma plot: {e}")


def verify_pointwise_error_bound_main(W_star_cpu, U_cpu, S_cpu, Vh_cpu,
                                      query_embeddings_all, passage_embeddings_all,
                                      qid_to_idx_map, pid_to_idx_map,
                                      ranks_to_check_errors, num_samples, results_dir):
    """Fixed: Verify pointwise error bound with proper GPU memory management"""
    logger = logging.getLogger('experiment3')

    if not exp3_config.VERIFY_POINTWISE_ERROR_BOUND:
        logger.info("Skipping pointwise error bound verification (disabled in config).")
        return

    logger.info(f"Verifying pointwise error bound with {num_samples} samples...")

    all_qids_list = list(qid_to_idx_map.keys())
    all_pids_list = list(pid_to_idx_map.keys())

    if not all_qids_list or not all_pids_list:
        logger.warning("Not enough query/passage IDs with embeddings to verify pointwise error.")
        return

    # Sample embeddings on CPU first
    sampled_q_embeds_list = []
    sampled_d_embeds_list = []
    sampled_q_norms_list = []
    sampled_d_norms_list = []

    for _ in range(num_samples):
        qid = random.choice(all_qids_list)
        pid = random.choice(all_pids_list)

        q_embed = torch.tensor(query_embeddings_all[qid_to_idx_map[qid]], dtype=torch.float32)
        d_embed = torch.tensor(passage_embeddings_all[pid_to_idx_map[pid]], dtype=torch.float32)

        sampled_q_embeds_list.append(q_embed)
        sampled_d_embeds_list.append(d_embed)
        sampled_q_norms_list.append(torch.linalg.norm(q_embed).item())
        sampled_d_norms_list.append(torch.linalg.norm(d_embed).item())

    error_data_list = []

    # Use GPU memory manager context
    with gpu_memory_manager():
        # Stack and move to device for batch operations
        sampled_q_embeds_dev = torch.stack(sampled_q_embeds_list).to(exp3_config.DEVICE)
        sampled_d_embeds_dev = torch.stack(sampled_d_embeds_list).to(exp3_config.DEVICE)
        W_star_dev = W_star_cpu.to(exp3_config.DEVICE)

        # Calculate s_W_star scores once
        with torch.no_grad():
            scores_W_star_dev = torch.sum((sampled_q_embeds_dev @ W_star_dev) * sampled_d_embeds_dev, dim=1)

        U_dev, S_dev, Vh_dev = U_cpu.to(exp3_config.DEVICE), S_cpu.to(exp3_config.DEVICE), Vh_cpu.to(exp3_config.DEVICE)

        for r_val in tqdm(ranks_to_check_errors, desc="Verifying Pointwise Error for Ranks"):
            if r_val >= len(S_dev):
                logger.warning(
                    f"Rank r={r_val} is too high for pointwise error check (max SVD rank {len(S_dev) - 1}). Skipping.")
                continue

            sigma_r_plus_1_val = S_dev[r_val].item()
            W_r_dev = construct_Wr_approximation(U_dev, S_dev, Vh_dev, r_val).to(exp3_config.DEVICE)

            with torch.no_grad():
                scores_W_r_dev = torch.sum((sampled_q_embeds_dev @ W_r_dev) * sampled_d_embeds_dev, dim=1)

            actual_errors_np = torch.abs(scores_W_star_dev - scores_W_r_dev).cpu().numpy()

            for i in range(num_samples):
                theoretical_bound_val = sigma_r_plus_1_val * sampled_q_norms_list[i] * sampled_d_norms_list[i]
                is_bound_held_flag = actual_errors_np[i] <= theoretical_bound_val + 1e-5
                error_data_list.append({
                    "rank_r": r_val, "sample_idx": i,
                    "actual_error": float(actual_errors_np[i]),
                    "theoretical_bound": float(theoretical_bound_val),
                    "sigma_r_plus_1": float(sigma_r_plus_1_val),
                    "q_norm": float(sampled_q_norms_list[i]),
                    "d_norm": float(sampled_d_norms_list[i]),
                    "is_bound_held": is_bound_held_flag
                })
                if not is_bound_held_flag:
                    logger.warning(
                        f"Bound NOT held for r={r_val}, sample={i}: actual={actual_errors_np[i]:.4f}, bound={theoretical_bound_val:.4f}")

            # Clean up intermediate GPU tensors
            del W_r_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save error data
    error_df_path = os.path.join(results_dir, "pointwise_error_verification.csv")
    try:
        df = pd.DataFrame(error_data_list)
        df.to_csv(error_df_path, index=False)
        logger.info(f"Pointwise error verification data saved to {error_df_path}")

        violations = df[~df["is_bound_held"]]
        logger.info(f"Total pointwise error samples checked: {len(df)}")
        logger.info(f"Number of bound violations: {len(violations)}")
        if not violations.empty:
            logger.warning(f"Found {len(violations)} pointwise error bound violations. Check CSV for details.")
        else:
            logger.info("Pointwise error bound held for all tested samples.")
    except Exception as e:
        logger.error(f"Could not save pointwise error to CSV due to: {e}. Saving as JSON.")
        json_path = os.path.join(results_dir, "pointwise_error_verification.json")
        with open(json_path, 'w') as f:
            json.dump(error_data_list, f, indent=2)
        logger.info(f"Pointwise error verification data saved to {json_path}")


def determine_primary_metric(dataset_name):
    """Fixed: Centralized metric determination with proper defaults"""
    metric_map = {
        "car": "map",
        "robust": "ndcg_cut_10",
        "msmarco": "mrr_cut_10",
        "msmarco-passage": "mrr_cut_10"
    }
    return metric_map.get(dataset_name, "mrr_cut_10")  # Default to MRR@10


def get_qrels_path_for_dataset(dataset_name):
    """Get appropriate qrels path based on dataset"""
    if dataset_name == "car":
        return getattr(exp2_config, 'CAR_QRELS_FILE', None)
    elif dataset_name == "robust":
        return getattr(exp2_config, 'ROBUST_QRELS_FILE', None)
    else:
        return None  # MS MARCO uses ir_datasets


import argparse


def parse_arguments():
    """Parse command line arguments for Experiment 3"""
    parser = argparse.ArgumentParser(description='Experiment 3: Low-Rank Approximation Analysis')

    # Model and dataset arguments
    parser.add_argument('--model-path', type=str,
                        default=getattr(exp3_config, 'PRETRAINED_W_STAR_MODEL_PATH', None),
                        help='Path to the pretrained W* model file')
    parser.add_argument('--model-key', type=str,
                        default=getattr(exp3_config, 'PRETRAINED_W_STAR_MODEL_KEY', 'full_rank_bilinear'),
                        help='Model key/type (e.g., full_rank_bilinear, low_rank_bilinear)')
    parser.add_argument('--dataset', type=str,
                        default=getattr(exp3_config, 'DATASET_NAME', 'msmarco'),
                        choices=['msmarco', 'car', 'robust'],
                        help='Dataset to use for evaluation')

    # Rank testing arguments
    parser.add_argument('--ranks', type=int, nargs='+',
                        default=getattr(exp3_config, 'EXP3_RANKS_TO_TEST', [1, 2, 4, 8, 16, 32, 64, 128]),
                        help='List of ranks to test (e.g., --ranks 1 2 4 8 16 32)')
    parser.add_argument('--max-rank', type=int, default=None,
                        help='Maximum rank to test (will filter provided ranks)')

    # Output arguments
    parser.add_argument('--results-dir', type=str,
                        default=getattr(exp3_config, 'EXP3_RESULTS_DIR', 'results/experiment3'),
                        help='Directory to save results')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name to append to results directory')

    # Evaluation arguments
    parser.add_argument('--verify-bounds', action='store_true',
                        default=getattr(exp3_config, 'VERIFY_POINTWISE_ERROR_BOUND', True),
                        help='Verify pointwise error bounds')
    parser.add_argument('--num-samples', type=int,
                        default=getattr(exp3_config, 'NUM_POINTWISE_ERROR_SAMPLES', 1000),
                        help='Number of samples for pointwise error verification')

    # Device argument
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device to use (cpu, cuda, or auto)')

    # Control arguments
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Only do SVD analysis, skip evaluation')

    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration based on command line arguments"""
    logger = logging.getLogger('experiment3')

    # Update paths
    if args.model_path:
        exp3_config.PRETRAINED_W_STAR_MODEL_PATH = args.model_path
        logger.info(f"Using model path from command line: {args.model_path}")

    if args.model_key:
        exp3_config.PRETRAINED_W_STAR_MODEL_KEY = args.model_key

    if args.dataset:
        exp3_config.DATASET_NAME = args.dataset

    # Update ranks
    if args.ranks:
        exp3_config.EXP3_RANKS_TO_TEST = sorted(list(set(args.ranks)))
        logger.info(f"Using ranks from command line: {exp3_config.EXP3_RANKS_TO_TEST}")

    # Update results directory
    if args.run_name:
        base_dir = args.results_dir
        exp3_config.EXP3_RESULTS_DIR = os.path.join(base_dir, args.run_name)
        logger.info(f"Using results directory: {exp3_config.EXP3_RESULTS_DIR}")
    else:
        exp3_config.EXP3_RESULTS_DIR = args.results_dir

    # Update verification settings
    exp3_config.VERIFY_POINTWISE_ERROR_BOUND = args.verify_bounds
    exp3_config.NUM_POINTWISE_ERROR_SAMPLES = args.num_samples

    # Update device
    if args.device:
        if args.device == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        exp3_config.DEVICE = device
        logger.info(f"Using device: {device}")

    return args


def main():
    """Fixed main function with command line argument support"""
    # Parse command line arguments first
    args = parse_arguments()

    if not getattr(exp3_config, 'EXP3_ENABLED', True):
        print("Experiment 3 is disabled in experiment3/config.py. Use --help for options.")
        return

    logger = setup_logging_exp3(exp3_config.EXP3_RESULTS_DIR)

    # Update config based on command line arguments
    args = update_config_from_args(args)

    logger.info("Starting Experiment 3: Low-Rank Approximation Analysis")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info(f"Using device: {exp3_config.DEVICE}")
    logger.info(f"Model path: {exp3_config.PRETRAINED_W_STAR_MODEL_PATH}")
    logger.info(f"Model key: {exp3_config.PRETRAINED_W_STAR_MODEL_KEY}")
    logger.info(f"Dataset: {exp3_config.DATASET_NAME}")
    logger.info(f"Ranks to test: {exp3_config.EXP3_RANKS_TO_TEST}")

    # Validate required arguments
    if not exp3_config.PRETRAINED_W_STAR_MODEL_PATH:
        logger.error("No model path specified. Use --model_path or set in config.")
        return

    # Determine dataset and data loading strategy (Fixed)
    dataset_name, use_ir_datasets = determine_dataset_info(exp3_config.PRETRAINED_W_STAR_MODEL_PATH)
    logger.info(f"Detected dataset: {dataset_name}, using ir_datasets: {use_ir_datasets}")

    # Determine primary metric for this dataset
    primary_metric = determine_primary_metric(dataset_name)
    logger.info(f"Primary metric for {dataset_name}: {primary_metric}")

    # --- 1. Load W* and Perform SVD ---
    try:
        W_star_cpu = get_W_star_matrix(
            exp3_config.PRETRAINED_W_STAR_MODEL_PATH,
            exp3_config.PRETRAINED_W_STAR_MODEL_KEY,
            exp2_config.MODEL_CONFIGS
        )
        logger.info(f"Successfully loaded W* matrix with shape: {W_star_cpu.shape}")
    except Exception as e:
        logger.error(f"Failed to load W_star: {e}. Exiting Experiment 3.")
        import traceback
        logger.error(traceback.format_exc())
        return

    try:
        U_cpu, S_singular_values_cpu, Vh_cpu = perform_svd(W_star_cpu)
        if not args.skip_plots:
            plot_singular_values(S_singular_values_cpu, exp3_config.EXP3_RESULTS_DIR)
    except Exception as e:
        logger.error(f"Failed to perform SVD: {e}. Exiting Experiment 3.")
        return

    # If only SVD analysis requested, stop here
    if args.skip_evaluation:
        logger.info("Skipping evaluation as requested (--skip_evaluation)")
        logger.info("SVD analysis completed.")
        return

    # --- 2. Load Data for Evaluation ---
    logger.info(f"Loading embeddings and dev data for {dataset_name}...")

    try:
        # Fixed: Simplified dataset name handling
        dataset_for_loader = None if dataset_name == "msmarco" else dataset_name
        query_embeddings_all, passage_embeddings_all, qid_to_idx_map, pid_to_idx_map = load_embeddings_and_mappings(
            dataset_name=dataset_for_loader
        )
        logger.info(f"Loaded embeddings: {len(qid_to_idx_map)} queries, {len(pid_to_idx_map)} passages")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return

    try:
        # Load dev data using experiment2's method with dataset-specific logic
        dev_query_to_candidates_map = load_dev_data_for_eval(
            qid_to_idx_map,
            pid_to_idx_map,
            use_ir_datasets=use_ir_datasets,
            dataset_name=dataset_name
        )
        logger.info(f"Loaded dev data for {len(dev_query_to_candidates_map)} queries")
    except Exception as e:
        logger.error(f"Failed to load dev data: {e}")
        return

    # --- 3. Validate and Filter Ranks ---
    # Fixed: Use actual matrix dimensions instead of config
    max_possible_rank_svd = len(S_singular_values_cpu)
    actual_embedding_dim = W_star_cpu.shape[0]  # Use actual matrix dimensions
    max_rank_practical = min(max_possible_rank_svd, actual_embedding_dim)

    # Apply max_rank filter if specified
    if args.max_rank:
        max_rank_practical = min(max_rank_practical, args.max_rank)
        logger.info(f"Applied max_rank filter: {args.max_rank}")

    logger.info(f"Matrix dimensions: {W_star_cpu.shape}")
    logger.info(f"SVD rank limit: {max_possible_rank_svd}")
    logger.info(f"Practical max rank: {max_rank_practical}")

    # Filter ranks to be valid
    valid_ranks_to_test = sorted(list(set(
        r for r in exp3_config.EXP3_RANKS_TO_TEST if 1 <= r <= max_rank_practical
    )))

    if not valid_ranks_to_test:
        logger.error(f"No valid ranks to test based on EXP3_RANKS_TO_TEST ({exp3_config.EXP3_RANKS_TO_TEST}) "
                     f"and max practical rank {max_rank_practical}. Exiting.")
        return

    logger.info(f"Will test approximations for ranks: {valid_ranks_to_test}")

    # --- 4. Evaluate W_r for various ranks r ---
    evaluated_ranks = []
    scores_list = []

    qrels_path = get_qrels_path_for_dataset(dataset_name)

    for r_val in tqdm(valid_ranks_to_test, desc="Evaluating W_r Approximations"):
        logger.info(f"\n--- Evaluating for Rank r = {r_val} ---")

        try:
            with gpu_memory_manager():
                W_r_cpu = construct_Wr_approximation(U_cpu, S_singular_values_cpu, Vh_cpu, r_val)

                # Use BilinearScorer with the fixed W_r matrix
                temp_model_Wr = BilinearScorer(W_r_cpu.to(exp3_config.DEVICE))

                run_file_path = os.path.join(exp3_config.EXP3_RESULTS_DIR, f"run.dev.Wr_rank{r_val}.txt")

                # Fixed: Evaluate the model with proper error handling
                try:
                    primary_score, all_metrics = evaluate_model_on_dev(
                        model=temp_model_Wr,
                        query_embeddings=query_embeddings_all,
                        passage_embeddings=passage_embeddings_all,
                        qid_to_idx=qid_to_idx_map,
                        pid_to_idx=pid_to_idx_map,
                        dev_query_to_candidates=dev_query_to_candidates_map,
                        run_file_path=run_file_path,
                        use_ir_datasets=use_ir_datasets,
                        qrels_path=qrels_path,
                        dataset_name=dataset_name
                    )

                    # Fixed: Proper metric selection logic
                    if primary_metric in all_metrics:
                        final_score = all_metrics[primary_metric]
                        logger.info(f"Using {primary_metric} from all_metrics: {final_score:.4f}")
                    else:
                        final_score = primary_score
                        logger.info(f"Using primary_score (fallback): {final_score:.4f}")
                        logger.warning(
                            f"Primary metric '{primary_metric}' not found in all_metrics. Available: {list(all_metrics.keys())}")

                    logger.info(f"Rank r={r_val}, {primary_metric.upper()}: {final_score:.4f}")
                    evaluated_ranks.append(r_val)
                    scores_list.append(final_score)

                except Exception as eval_error:
                    logger.error(f"Evaluation failed for rank {r_val}: {eval_error}")
                    continue

        except Exception as rank_error:
            logger.error(f"Failed to process rank {r_val}: {rank_error}")
            continue

    # --- 5. Plotting and Saving Results ---
    if evaluated_ranks:
        logger.info(f"Successfully evaluated {len(evaluated_ranks)} ranks")

        if not args.skip_plots:
            try:
                plot_performance_vs_rank(evaluated_ranks, scores_list, exp3_config.EXP3_RESULTS_DIR,
                                         metric_name=primary_metric.upper())
                plot_performance_vs_sigma_r_plus_1(evaluated_ranks, scores_list, S_singular_values_cpu,
                                                   exp3_config.EXP3_RESULTS_DIR,
                                                   metric_name=primary_metric.upper())
            except Exception as plot_error:
                logger.error(f"Plotting failed: {plot_error}")
        else:
            logger.info("Skipping plots as requested (--skip_plots)")

        # Save results summary
        try:
            results_summary = {
                "experiment_config": {
                    "PRETRAINED_W_STAR_MODEL_PATH": exp3_config.PRETRAINED_W_STAR_MODEL_PATH,
                    "PRETRAINED_W_STAR_MODEL_KEY": exp3_config.PRETRAINED_W_STAR_MODEL_KEY,
                    "EXP3_RANKS_TO_TEST_CONFIGURED": exp3_config.EXP3_RANKS_TO_TEST,
                    "dataset_name": dataset_name,
                    "use_ir_datasets": use_ir_datasets,
                    "primary_metric": primary_metric,
                    "W_star_shape": list(W_star_cpu.shape),
                    "max_rank_practical": max_rank_practical,
                    "command_line_args": vars(args)
                },
                "ranks_evaluated": evaluated_ranks,
                "scores": scores_list,
                "primary_metric": primary_metric,
                "singular_values_all": S_singular_values_cpu.tolist(),
                "summary_stats": {
                    "best_rank": evaluated_ranks[scores_list.index(max(scores_list))] if scores_list else None,
                    "best_score": max(scores_list) if scores_list else None,
                    "worst_score": min(scores_list) if scores_list else None,
                    "score_range": max(scores_list) - min(scores_list) if scores_list else None
                }
            }
            summary_file = os.path.join(exp3_config.EXP3_RESULTS_DIR, "experiment3_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(results_summary, f, indent=4)
            logger.info(f"Experiment 3 summary saved to {summary_file}")
        except Exception as save_error:
            logger.error(f"Failed to save results summary: {save_error}")
    else:
        logger.warning("No ranks were successfully evaluated. Skipping result summary and plots.")

    # --- 6. Optional: Verify Pointwise Error Bound ---
    if exp3_config.VERIFY_POINTWISE_ERROR_BOUND and evaluated_ranks:
        try:
            # Fixed: Better selection of ranks for pointwise check
            num_ranks_to_check = min(5, len(evaluated_ranks))
            if num_ranks_to_check > 0:
                step = max(1, len(evaluated_ranks) // num_ranks_to_check)
                ranks_for_pointwise_check = evaluated_ranks[::step]
                if evaluated_ranks[0] not in ranks_for_pointwise_check:
                    ranks_for_pointwise_check.insert(0, evaluated_ranks[0])

                logger.info(f"Selected ranks for pointwise error check: {ranks_for_pointwise_check}")

                verify_pointwise_error_bound_main(
                    W_star_cpu, U_cpu, S_singular_values_cpu, Vh_cpu,
                    query_embeddings_all, passage_embeddings_all,
                    qid_to_idx_map, pid_to_idx_map,
                    ranks_for_pointwise_check,
                    exp3_config.NUM_POINTWISE_ERROR_SAMPLES,
                    exp3_config.EXP3_RESULTS_DIR
                )
            else:
                logger.info("No ranks available for pointwise error check.")
        except Exception as pointwise_error:
            logger.error(f"Pointwise error verification failed: {pointwise_error}")
    else:
        logger.info("Skipping pointwise error check (disabled in config or no ranks evaluated).")

    logger.info("Experiment 3 finished successfully.")

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # Ensure matplotlib uses a non-interactive backend if running in a headless environment
    try:
        import matplotlib

        if 'DISPLAY' not in os.environ and os.name != 'nt':
            matplotlib.use('Agg')
            print("Matplotlib backend set to 'Agg' for headless environment.")
    except ImportError:
        print("Matplotlib not found. Plotting will fail.")

    try:
        main()
    except Exception as e:
        print(f"Experiment 3 failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)