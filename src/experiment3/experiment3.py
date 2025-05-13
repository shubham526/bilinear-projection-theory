# experiment3/main_experiment3.py

import torch
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
import pandas as pd  # For saving CSV

# --- Add project root to sys.path to allow importing from experiment2 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config as exp3_config

# Import necessary components from Experiment 2
sys.path.append(os.path.join(project_root, 'experiment2'))
from src.experiment2.models import get_model as get_exp2_model, FullRankBilinearModel, LowRankBilinearModel
from src.experiment2.data_loader import load_embeddings_and_mappings, load_dev_data_for_eval
from src.experiment2.evaluate import evaluate_model_on_dev
import src.experiment2.config as exp2_config
from src.experiment1.models import BilinearScorer



def setup_logging_exp3(results_dir):
    """Setup logging for Experiment 3"""
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, 'experiment3.log')

    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_W_star_matrix(model_path, model_key_from_exp2_config, exp2_model_configs):
    """
    Loads a pre-trained model and extracts its W* matrix.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading W* from model path: {model_path} (type: {model_key_from_exp2_config})")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_config_params = exp2_model_configs.get(model_key_from_exp2_config)
    if not model_config_params:
        logger.error(f"Model key '{model_key_from_exp2_config}' not found in Experiment 2 MODEL_CONFIGS.")
        raise ValueError(f"Model key '{model_key_from_exp2_config}' not found in Experiment 2 MODEL_CONFIGS.")

    # Instantiate model structure
    model_architecture = get_exp2_model(model_key_from_exp2_config, model_config_params)

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
    logger = logging.getLogger(__name__)
    logger.info("Performing SVD on W* matrix (on CPU)...")
    U, S_singular_values, Vh = torch.linalg.svd(W_star_matrix_cpu.float())
    logger.info(f"SVD completed. U shape: {U.shape}, S shape: {S_singular_values.shape}, Vh shape: {Vh.shape}")
    return U, S_singular_values, Vh


def construct_Wr_approximation(U, S_singular_values, Vh, r):
    """Constructs rank-r approximation W_r = U_r @ diag(S_r) @ Vh_r."""
    logger = logging.getLogger(__name__)
    if r < 1:
        raise ValueError("Rank r must be at least 1.")

    actual_r = min(r, S_singular_values.shape[0], U.shape[1], Vh.shape[0])
    if actual_r != r:
        logger.warning(f"Requested rank r={r} adjusted to actual_r={actual_r} due to matrix dimensions.")

    U_r = U[:, :actual_r]
    S_r_diag = torch.diag(S_singular_values[:actual_r])
    Vh_r = Vh[:actual_r, :]
    W_r = torch.matmul(U_r, torch.matmul(S_r_diag, Vh_r))
    return W_r


def plot_singular_values(singular_values, save_path_dir, filename="singular_values_spectrum.png"):
    logger = logging.getLogger(__name__)
    logger.info("Plotting singular value spectrum...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(singular_values) + 1), singular_values.numpy(), marker='o', linestyle='-')
    plt.yscale('log')
    plt.title('Singular Value Spectrum of W*')
    plt.xlabel('Singular Value Index (i)')
    plt.ylabel('Singular Value (σ_i) - Log Scale')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plot_file = os.path.join(save_path_dir, filename)
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"Singular value plot saved to {plot_file}")


def plot_performance_vs_rank(ranks_tested, mrr_scores, save_path_dir, filename="performance_vs_rank.png"):
    logger = logging.getLogger(__name__)
    logger.info("Plotting MRR@10 vs. Rank r...")
    plt.figure(figsize=(10, 6))
    plt.plot(ranks_tested, mrr_scores, marker='x', linestyle='-', linewidth=2, markersize=8)
    plt.title('Retrieval Performance (MRR@10) vs. Rank r of W_r')
    plt.xlabel('Rank (r)')
    plt.ylabel('MRR@10 on MS MARCO Dev')
    # Make x-ticks more readable
    if len(ranks_tested) > 10:
        plt.xticks(ranks_tested[::max(1, len(ranks_tested) // 10)] + [ranks_tested[-1]])
    else:
        plt.xticks(ranks_tested)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = os.path.join(save_path_dir, filename)
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"Performance vs. Rank plot saved to {plot_file}")


def plot_performance_vs_sigma_r_plus_1(ranks_tested, mrr_scores, singular_values, save_path_dir,
                                       filename="performance_vs_sigma_r_plus_1.png"):
    logger = logging.getLogger(__name__)
    logger.info("Plotting MRR@10 vs. σ_{r+1}...")

    sigma_r_plus_1_values = []
    valid_mrr_for_sigma_plot = []

    for i, r_val in enumerate(ranks_tested):
        if r_val < len(singular_values):
            sigma_r_plus_1_values.append(singular_values[r_val].item())
            valid_mrr_for_sigma_plot.append(mrr_scores[i])

    if not sigma_r_plus_1_values:
        logger.warning("Not enough data to plot performance vs sigma_r+1.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(sigma_r_plus_1_values, valid_mrr_for_sigma_plot, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title('Retrieval Performance (MRR@10) vs. Next Neglected Singular Value (σ_{r+1})')
    plt.xlabel('Next Singular Value (σ_{r+1}) - Log Scale')
    plt.ylabel('MRR@10 on MS MARCO Dev')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    # Invert x-axis for better visualization
    if len(sigma_r_plus_1_values) > 1 and sigma_r_plus_1_values[0] > sigma_r_plus_1_values[-1]:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plot_file = os.path.join(save_path_dir, filename)
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"Performance vs. Sigma_r+1 plot saved to {plot_file}")


def verify_pointwise_error_bound_main(W_star_cpu, U_cpu, S_cpu, Vh_cpu,
                                      query_embeddings_all, passage_embeddings_all,
                                      qid_to_idx_map, pid_to_idx_map,
                                      ranks_to_check_errors, num_samples, results_dir):
    logger = logging.getLogger(__name__)
    if not exp3_config.VERIFY_POINTWISE_ERROR_BOUND:
        logger.info("Skipping pointwise error bound verification (disabled in config).")
        return

    logger.info(f"Verifying pointwise error bound with {num_samples} samples...")

    all_qids_list = list(qid_to_idx_map.keys())
    all_pids_list = list(pid_to_idx_map.keys())

    if not all_qids_list or not all_pids_list:
        logger.warning("Not enough query/passage IDs with embeddings to verify pointwise error.")
        return

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

    # Stack and move to device for batch operations
    sampled_q_embeds_dev = torch.stack(sampled_q_embeds_list).to(exp3_config.DEVICE)
    sampled_d_embeds_dev = torch.stack(sampled_d_embeds_list).to(exp3_config.DEVICE)
    W_star_dev = W_star_cpu.to(exp3_config.DEVICE)

    # Calculate s_W_star scores once
    with torch.no_grad():
        scores_W_star_dev = torch.sum((sampled_q_embeds_dev @ W_star_dev) * sampled_d_embeds_dev, dim=1)

    error_data_list = []
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


def main():
    if not exp3_config.EXP3_ENABLED:
        print("Experiment 3 is disabled in experiment3/config.py. Skipping.")
        return

    logger = setup_logging_exp3(exp3_config.EXP3_RESULTS_DIR)
    logger.info("Starting Experiment 3: Low-Rank Approximation Analysis")
    logger.info(f"Using device: {exp3_config.DEVICE}")
    logger.info(
        f"Loading W* from: {exp3_config.PRETRAINED_W_STAR_MODEL_PATH} (type: {exp3_config.PRETRAINED_W_STAR_MODEL_KEY})")

    # --- 1. Load W* and Perform SVD ---
    try:
        W_star_cpu = get_W_star_matrix(
            exp3_config.PRETRAINED_W_STAR_MODEL_PATH,
            exp3_config.PRETRAINED_W_STAR_MODEL_KEY,
            exp2_config.MODEL_CONFIGS
        )
    except Exception as e:
        logger.error(f"Failed to load W_star: {e}. Exiting Experiment 3.")
        import traceback
        logger.error(traceback.format_exc())
        return

    U_cpu, S_singular_values_cpu, Vh_cpu = perform_svd(W_star_cpu)
    plot_singular_values(S_singular_values_cpu, exp3_config.EXP3_RESULTS_DIR)

    # --- 2. Load Data for Evaluation ---
    logger.info("Loading MS MARCO embeddings and dev data for evaluation...")
    query_embeddings_all, passage_embeddings_all, qid_to_idx_map, pid_to_idx_map = load_embeddings_and_mappings(
        query_emb_path=exp2_config.QUERY_EMBEDDINGS_PATH,
        passage_emb_path=exp2_config.PASSAGE_EMBEDDINGS_PATH,
        qid_map_path=exp2_config.QUERY_ID_TO_IDX_PATH,
        pid_map_path=exp2_config.PASSAGE_ID_TO_IDX_PATH
    )
    dev_query_to_candidates_map = load_dev_data_for_eval(
        dev_queries_path=exp2_config.DEV_QUERIES_PATH,
        dev_candidates_path=exp2_config.DEV_CANDIDATES_PATH,
        qid_to_idx=qid_to_idx_map,
        pid_to_idx=pid_to_idx_map
    )

    # --- 3. Evaluate W_r for various ranks r ---
    evaluated_ranks = []
    mrr_scores_list = []

    max_possible_rank_svd = len(S_singular_values_cpu)
    embedding_dim_from_exp2 = exp2_config.EMBEDDING_DIM
    max_rank_practical = min(max_possible_rank_svd, embedding_dim_from_exp2)

    # Filter ranks to be valid
    valid_ranks_to_test_config = sorted(list(set(
        r for r in exp3_config.EXP3_RANKS_TO_TEST if 1 <= r <= max_rank_practical
    )))

    if not valid_ranks_to_test_config:
        logger.error(f"No valid ranks to test based on EXP3_RANKS_TO_TEST ({exp3_config.EXP3_RANKS_TO_TEST}) "
                     f"and max practical rank {max_rank_practical}. Exiting.")
        return

    logger.info(f"Will test approximations for ranks: {valid_ranks_to_test_config}")

    for r_val in tqdm(valid_ranks_to_test_config, desc="Evaluating W_r Approximations"):
        logger.info(f"\n--- Evaluating for Rank r = {r_val} ---")
        W_r_cpu = construct_Wr_approximation(U_cpu, S_singular_values_cpu, Vh_cpu, r_val)

        # Use BilinearScorer with the fixed W_r matrix
        temp_model_Wr = BilinearScorer(W_r_cpu.to(exp3_config.DEVICE))

        run_file_path = os.path.join(exp3_config.EXP3_RESULTS_DIR, f"run.dev.Wr_rank{r_val}.txt")

        # Evaluate the model
        mrr_at_10 = evaluate_model_on_dev(
            model=temp_model_Wr,
            query_embeddings=query_embeddings_all,
            passage_embeddings=passage_embeddings_all,
            qid_to_idx=qid_to_idx_map,
            pid_to_idx=pid_to_idx_map,
            dev_query_to_candidates=dev_query_to_candidates_map,
            run_file_path=run_file_path,
        )
        logger.info(f"Rank r={r_val}, MRR@10: {mrr_at_10:.4f}")
        evaluated_ranks.append(r_val)
        mrr_scores_list.append(mrr_at_10)

    # --- 4. Plotting and Saving Results ---
    if evaluated_ranks:
        plot_performance_vs_rank(evaluated_ranks, mrr_scores_list, exp3_config.EXP3_RESULTS_DIR)
        plot_performance_vs_sigma_r_plus_1(evaluated_ranks, mrr_scores_list, S_singular_values_cpu,
                                           exp3_config.EXP3_RESULTS_DIR)

        results_summary = {
            "experiment_config": {
                "PRETRAINED_W_STAR_MODEL_PATH": exp3_config.PRETRAINED_W_STAR_MODEL_PATH,
                "PRETRAINED_W_STAR_MODEL_KEY": exp3_config.PRETRAINED_W_STAR_MODEL_KEY,
                "EXP3_RANKS_TO_TEST_CONFIGURED": exp3_config.EXP3_RANKS_TO_TEST,
            },
            "ranks_evaluated": evaluated_ranks,
            "mrr_scores": mrr_scores_list,
            "singular_values_all": S_singular_values_cpu.tolist()
        }
        summary_file = os.path.join(exp3_config.EXP3_RESULTS_DIR, "experiment3_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        logger.info(f"Experiment 3 summary saved to {summary_file}")
    else:
        logger.warning("No ranks were evaluated. Skipping result summary and plots.")

    # --- 5. Optional: Verify Pointwise Error Bound ---
    ranks_for_pointwise_check = [r for r in valid_ranks_to_test_config if
                                 r % (max(1, len(valid_ranks_to_test_config) // 5)) == 0 or r == 1]
    if not ranks_for_pointwise_check and valid_ranks_to_test_config:
        ranks_for_pointwise_check = [valid_ranks_to_test_config[0]]

    if ranks_for_pointwise_check and exp3_config.VERIFY_POINTWISE_ERROR_BOUND:
        verify_pointwise_error_bound_main(W_star_cpu, U_cpu, S_singular_values_cpu, Vh_cpu,
                                          query_embeddings_all, passage_embeddings_all,
                                          qid_to_idx_map, pid_to_idx_map,
                                          ranks_for_pointwise_check,
                                          exp3_config.NUM_POINTWISE_ERROR_SAMPLES,
                                          exp3_config.EXP3_RESULTS_DIR)
    else:
        logger.info("Skipping pointwise error check (no ranks selected or VERIFY_POINTWISE_ERROR_BOUND is False).")

    logger.info("Experiment 3 finished.")


if __name__ == "__main__":
    # Ensure matplotlib uses a non-interactive backend if running in a headless environment
    try:
        import matplotlib

        if 'DISPLAY' not in os.environ and os.name != 'nt':
            matplotlib.use('Agg')
            print("Matplotlib backend set to 'Agg' for headless environment.")
    except ImportError:
        print("Matplotlib not found. Plotting will fail.")

    main()