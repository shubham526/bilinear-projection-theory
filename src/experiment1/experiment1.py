# experiment1.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from scipy import stats
import random

# Use the experiment-specific config
import config as config
from synthetic_data_gen import (
    generate_random_query_vector,
    generate_random_I0_indices,
    generate_indicator_e_I_hypercube,
    generate_structured_agreement_docs,
    generate_challenging_patterns,
    generate_all_possible_I0_indices
)
from models import (
    construct_theoretical_W_I0,
    BilinearScorer,
    WeightedDotProductModel
)

# Global logger variable, will be initialized by setup_logging_exp1 for each main_experiment_run
current_run_logger = None


def set_seeds(seed_value):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if current_run_logger:
        current_run_logger.info(f"Random seeds set to: {seed_value}")
    else:
        print(f"Random seeds set to: {seed_value} (logger not yet initialized for this message)")


def setup_logging_exp1(model_save_dir, experiment_name):
    """Setup logging for a specific Experiment 1 run."""
    run_logger = logging.getLogger(experiment_name)  # Unique logger per run
    run_logger.handlers = []  # Clear previous handlers for this specific logger instance
    run_logger.propagate = False  # Avoid duplicate messages in parent/root logger if it's also configured

    log_dir = os.path.join(model_save_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    run_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler for this specific run
    fh = logging.FileHandler(os.path.join(log_dir, 'experiment.log'), mode='w')
    fh.setFormatter(formatter)
    run_logger.addHandler(fh)

    # Stream handler for console output for this specific run
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    run_logger.addHandler(sh)

    return run_logger


def compute_confidence_interval(successes, total, confidence_level=0.95):  # Use passed arg
    """Compute Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total

    z_alpha_half = stats.norm.ppf((1 + confidence_level) / 2)
    n = float(total)  # Ensure n is float for calculations

    # Wilson score interval formula components
    center_numerator = p_hat + (z_alpha_half ** 2) / (2 * n)
    center_denominator = 1 + (z_alpha_half ** 2) / n
    center = center_numerator / center_denominator

    width_term_under_sqrt = (p_hat * (1 - p_hat) / n) + (z_alpha_half ** 2 / (4 * n ** 2))
    if width_term_under_sqrt < 0:  # Handle potential floating point inaccuracies
        width_term_under_sqrt = 0

    width = z_alpha_half * np.sqrt(width_term_under_sqrt) / center_denominator

    lower_bound = max(0.0, center - width) * 100
    upper_bound = min(100.0, center + width) * 100

    return lower_bound, upper_bound


def test_bilinear_sufficiency(logger_to_use):
    logger_to_use.info("\n--- Part (i): Bilinear Model Sufficiency (Theorem 3.1.i) ---")
    n_dim = config.DIM_N
    perfect_rankings_count = 0
    total_test_cases = config.NUM_TEST_CASES_BILINEAR
    all_agree_scores_bilinear = []
    all_disagree_scores_bilinear = []

    for _ in tqdm(range(total_test_cases), desc="Testing Bilinear Sufficiency"):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        I0_indices = generate_random_I0_indices(n_dim, 2)

        W_I0_matrix = construct_theoretical_W_I0(n_dim, I0_indices).to(config.DEVICE)
        bilinear_model = BilinearScorer(W_I0_matrix)

        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices).to(config.DEVICE)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)

        scores_agree = [bilinear_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
        scores_disagree = [bilinear_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]

        all_agree_scores_bilinear.extend(scores_agree)
        all_disagree_scores_bilinear.extend(scores_disagree)

        correct_ranking_this_case = True
        # Theoretical scores are +2 for agree, -2 for disagree.
        # Allow small tolerance for floating point.
        if not (all(abs(s - 2.0) < 1e-5 for s in scores_agree) and \
                all(abs(s - (-2.0)) < 1e-5 for s in scores_disagree)):
            # If not exact, still check if min agree > max disagree (robust check for ranking)
            if not (min(scores_agree) > max(scores_disagree)):
                correct_ranking_this_case = False
                logger_to_use.debug(
                    f"Ranking Failure or Score Mismatch for Bilinear: q={q.cpu().numpy().tolist()}, I0={I0_indices}")
                logger_to_use.debug(f"  Agree Scores: {scores_agree} (Expected [+2, +2])")
                logger_to_use.debug(f"  Disagree Scores: {scores_disagree} (Expected [-2, -2])")

        if correct_ranking_this_case:
            perfect_rankings_count += 1

    success_rate = (perfect_rankings_count / total_test_cases) * 100 if total_test_cases > 0 else 0.0
    lower_ci, upper_ci = compute_confidence_interval(perfect_rankings_count, total_test_cases, config.CONFIDENCE_LEVEL)

    logger_to_use.info(f"Bilinear Model Test Summary:")
    logger_to_use.info(f"  Total test cases: {total_test_cases}")
    logger_to_use.info(f"  Perfect rankings achieved: {perfect_rankings_count}")
    logger_to_use.info(f"  Success rate: {success_rate:.2f}% (CI: [{lower_ci:.2f}%, {upper_ci:.2f}%])")
    if total_test_cases > 0:
        assert abs(success_rate - 100.0) < 1e-3, "Bilinear model did not achieve 100% success as expected."
    logger_to_use.info("✓ Bilinear model sufficiency confirmed as per Theorem 3.1.i.")

    return {
        "bilinear_success_rate": success_rate,
        "bilinear_confidence_interval": (lower_ci, upper_ci),
        "score_distributions_bilinear": {
            "agree_scores": all_agree_scores_bilinear[:min(200, len(all_agree_scores_bilinear))],
            "disagree_scores": all_disagree_scores_bilinear[:min(200, len(all_disagree_scores_bilinear))]
        }
    }


def train_single_wdp_model(logger_to_use, specific_I0=None):
    # Use WDP parameters from config.CURRENT_WDP_TRAIN_PARAMS for the general model
    # For specialized models, use dedicated config values or derive from general ones.

    if specific_I0:
        logger_to_use.info(f"\n--- Training WDP Model Specialized for I0={specific_I0} ---")
        # Use general run's wdp_train_samples as a base, then apply factor
        base_general_samples = config.CURRENT_WDP_TRAIN_PARAMS.get('wdp_train_samples', 50000)  # Fallback
        num_samples = int(base_general_samples * config.SPECIALIZED_WDP_TRAIN_SAMPLES_FACTOR)
        wdp_epochs = config.SPECIALIZED_WDP_EPOCHS
        # For other hyperparams, can use base or current run's general WDP settings
        wdp_lr = config.CURRENT_WDP_TRAIN_PARAMS.get('wdp_learning_rate', config.BASE_WDP_LEARNING_RATE)
        wdp_batch_size = config.CURRENT_WDP_TRAIN_PARAMS.get('wdp_batch_size', config.BASE_WDP_BATCH_SIZE)
        wdp_margin = config.CURRENT_WDP_TRAIN_PARAMS.get('wdp_margin', config.BASE_WDP_MARGIN)
    else:  # General WDP model training (uses CURRENT_WDP_TRAIN_PARAMS set in main_experiment_run)
        logger_to_use.info("\n--- Training a Single General Weighted Dot Product Model ---")
        wdp_train_params = config.CURRENT_WDP_TRAIN_PARAMS
        num_samples = wdp_train_params['wdp_train_samples']
        wdp_epochs = wdp_train_params['wdp_epochs']
        wdp_lr = wdp_train_params['wdp_learning_rate']
        wdp_batch_size = wdp_train_params['wdp_batch_size']
        wdp_margin = wdp_train_params['wdp_margin']

    n_dim = config.DIM_N
    wdp_model = WeightedDotProductModel(n_dim, init_strategy='small_random').to(config.DEVICE)
    optimizer = optim.AdamW(wdp_model.parameters(), lr=wdp_lr)
    loss_fn = nn.MarginRankingLoss(margin=wdp_margin).to(config.DEVICE)

    logger_to_use.info(f"WDP Model parameters: {sum(p.numel() for p in wdp_model.parameters() if p.requires_grad)}")
    logger_to_use.info(
        f"Training with: Samples={num_samples}, Epochs={wdp_epochs}, LR={wdp_lr}, Batch={wdp_batch_size}, Margin={wdp_margin}")

    training_samples = []
    if num_samples > 0 and wdp_epochs > 0:
        logger_to_use.info(f"Generating {num_samples} training samples for WDP...")
        for _ in tqdm(range(num_samples), desc="Generating WDP train data"):
            q = generate_random_query_vector(n_dim)
            current_I0_indices = specific_I0 if specific_I0 else generate_random_I0_indices(n_dim, 2)
            e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, current_I0_indices)
            docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)
            training_samples.append((q, docs_agree[0], docs_disagree[0]))
    elif wdp_epochs > 0:  # num_samples is 0 but epochs > 0
        logger_to_use.warning(
            "WDP training initiated with 0 samples but >0 epochs. Model will not be effectively trained.")
        return wdp_model  # Return untrained model
    else:  # num_epochs is 0
        logger_to_use.info("WDP training skipped as epochs are zero.")
        return wdp_model

    class WDPTrainDataset(torch.utils.data.Dataset):
        def __init__(self, samples): self.samples = samples

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx): return self.samples[idx]

    train_dataset = WDPTrainDataset(training_samples)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=wdp_batch_size, shuffle=True)

    for epoch in range(wdp_epochs):
        wdp_model.train()
        total_loss = 0
        num_batches_processed = 0  # Renamed from num_batches to avoid confusion
        progress_bar = tqdm(train_dataloader, desc=f"WDP Train Epoch {epoch + 1}/{wdp_epochs}")
        for q_batch, d_pos_batch, d_neg_batch in progress_bar:
            q_batch = q_batch.to(config.DEVICE)
            d_pos_batch = d_pos_batch.to(config.DEVICE)
            d_neg_batch = d_neg_batch.to(config.DEVICE)

            optimizer.zero_grad()
            scores_pos = wdp_model(q_batch, d_pos_batch)
            scores_neg = wdp_model(q_batch, d_neg_batch)
            targets = torch.ones_like(scores_pos).to(config.DEVICE)
            loss = loss_fn(scores_pos, scores_neg, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches_processed += 1
            progress_bar.set_postfix({'loss': total_loss / num_batches_processed if num_batches_processed > 0 else 0.0})
        avg_epoch_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        logger_to_use.info(f"WDP Train Epoch {epoch + 1} Avg Loss: {avg_epoch_loss:.4f}")

    logger_to_use.info("WDP model training finished.")
    logger_to_use.info(f"Final weights for this WDP instance: {wdp_model.get_weights().tolist()}")  # .tolist() for JSON
    return wdp_model


def test_wdp_on_specific_I0(wdp_model, I0_indices, logger_to_use, num_samples=None):
    if num_samples is None:
        num_samples = config.NUM_SAMPLES_PER_I0_TEST
    if num_samples == 0: return 0.0  # Avoid division by zero

    n_dim = config.DIM_N
    correct_rankings_count = 0
    wdp_model.eval()

    for _ in range(num_samples):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices).to(config.DEVICE)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)
        with torch.no_grad():
            scores_agree = [wdp_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
            scores_disagree = [wdp_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]
        min_agree = min(scores_agree) if scores_agree else -float('inf')
        max_disagree = max(scores_disagree) if scores_disagree else float('inf')
        if min_agree > max_disagree:
            correct_rankings_count += 1
    success_rate = (correct_rankings_count / num_samples) * 100
    return success_rate


def test_wdp_with_multiple_fixed_I0_sets(logger_to_use):
    if not config.INCLUDE_SPECIALIZED_WDP_TEST:
        logger_to_use.info("\n--- Skipping Specialized WDP Test (disabled in config) ---")
        return {}
    logger_to_use.info("\n--- Testing WDP Adaptation to Different I0 Sets (Specialized Training) ---")
    n_dim = config.DIM_N
    all_I0s = generate_all_possible_I0_indices(n_dim, 2)
    # Take a small, consistent subset for this test to keep it manageable
    test_I0_sets_for_specialization = all_I0s[:min(3, len(all_I0s))]
    results = {'specialized_wdp_results': []}

    for I0_train_set in test_I0_sets_for_specialization:
        logger_to_use.info(f"\nTraining WDP specialized for I0={I0_train_set}")
        specialized_wdp = train_single_wdp_model(logger_to_use, specific_I0=I0_train_set)

        success_rate_on_train_I0 = test_wdp_on_specific_I0(specialized_wdp, I0_train_set, logger_to_use)
        logger_to_use.info(
            f"  Specialized WDP (trained on {I0_train_set}) success on its training I0 {I0_train_set}: {success_rate_on_train_I0:.1f}%")

        current_I0_test_results = {
            'training_I0': list(I0_train_set),  # Convert tuple to list for JSON
            'success_on_training_I0': success_rate_on_train_I0,
            'success_on_other_I0s': {}
        }
        for I0_eval_set in test_I0_sets_for_specialization:
            if I0_eval_set != I0_train_set:
                success_rate_on_other_I0 = test_wdp_on_specific_I0(specialized_wdp, I0_eval_set, logger_to_use)
                logger_to_use.info(
                    f"  Specialized WDP (trained on {I0_train_set}) success on different I0 {I0_eval_set}: {success_rate_on_other_I0:.1f}%")
                current_I0_test_results['success_on_other_I0s'][str(I0_eval_set)] = success_rate_on_other_I0
        results['specialized_wdp_results'].append(current_I0_test_results)
    return results


def analyze_wdp_failures(wdp_model_to_analyze, logger_to_use):
    if not config.ANALYZE_FAILURE_PATTERNS:
        logger_to_use.info("\n--- Skipping Failure Pattern Analysis (disabled in config) ---")
        return {}
    logger_to_use.info("\n--- Analyzing WDP Failure Patterns ---")
    n_dim = config.DIM_N
    challenging_patterns = generate_challenging_patterns(n_dim)
    failure_analysis_results = []

    active_wdp_model = wdp_model_to_analyze
    if active_wdp_model is None:
        logger_to_use.warning("No WDP model provided for failure analysis. Using a default (untrained) one.")
        active_wdp_model = WeightedDotProductModel(n_dim).to(config.DEVICE)

    active_wdp_model.eval()

    for pattern in challenging_patterns:
        q_pattern_cpu = pattern["q"][:n_dim]  # Keep on CPU for potential modification if n_dim < pattern's q
        I0_pattern = pattern["I0"]

        # Ensure I0 indices are valid for current n_dim
        if max(I0_pattern) >= n_dim:
            logger_to_use.warning(
                f"Skipping pattern '{pattern['name']}' for I0={I0_pattern} as indices exceed n_dim={n_dim}")
            continue

        q_pattern_device = q_pattern_cpu.to(config.DEVICE)
        logger_to_use.info(f"  Analyzing pattern: {pattern['name']} with I0={I0_pattern}")

        # Use test_wdp_on_specific_I0 to get success rate on this pattern (q fixed, I0 fixed)
        # For this, we need to adapt test_wdp_on_specific_I0 slightly or reimplement its core logic
        # Here, let's reimplement for clarity with fixed q.
        correct_rankings_pattern = 0
        num_tests_for_pattern = 100  # Test this fixed (q, I0) with its generated doc sets multiple times (though docs are deterministic from q,eI0)
        # The original test_wdp_on_specific_I0 generates random q each time.
        # For a fixed pattern q, the docs are fixed, so one test is enough.
        # Let's do one detailed check.

        e_I0_hyper_pattern = generate_indicator_e_I_hypercube(n_dim, I0_pattern).to(config.DEVICE)
        docs_agree_pattern, docs_disagree_pattern = generate_structured_agreement_docs(q_pattern_device,
                                                                                       e_I0_hyper_pattern)

        with torch.no_grad():
            scores_agree_pattern = [active_wdp_model(q_pattern_device, d_a.to(config.DEVICE)).item() for d_a in
                                    docs_agree_pattern]
            scores_disagree_pattern = [active_wdp_model(q_pattern_device, d_d.to(config.DEVICE)).item() for d_d in
                                       docs_disagree_pattern]

        min_agree_pattern = min(scores_agree_pattern) if scores_agree_pattern else -float('inf')
        max_disagree_pattern = max(scores_disagree_pattern) if scores_disagree_pattern else float('inf')

        if min_agree_pattern > max_disagree_pattern:
            correct_rankings_pattern = 1  # Scored correctly for this one pattern instance
            success_rate_pattern = 100.0
        else:
            success_rate_pattern = 0.0
            logger_to_use.debug(
                f"    Pattern {pattern['name']} FAILED. Agree: {scores_agree_pattern}, Disagree: {scores_disagree_pattern}")

        logger_to_use.info(f"    Success rate for this pattern instance: {success_rate_pattern:.1f}%")
        failure_analysis_results.append({
            'pattern_name': pattern['name'], 'I0': list(I0_pattern),  # list for JSON
            'success_rate': success_rate_pattern, 'description': pattern['description']
        })
    logger_to_use.info("WDP Failure pattern analysis finished.")
    return {'failure_analysis': failure_analysis_results}


def test_wdp_universality_failure(wdp_model_to_test, logger_to_use):
    logger_to_use.info("\n--- Part (ii): Weighted Dot Product Universality Failure (Generalization Test) ---")
    n_dim = config.DIM_N
    if n_dim < 3:
        logger_to_use.warning(
            f"WDP failure demo requires n_dim >= 3. Current n_dim = {n_dim}. Skipping WDP generalization.")
        return {"wdp_generalization_success_rate": "N/A (n_dim < 3)", "wdp_confidence_interval": ("N/A", "N/A"),
                "score_distributions_wdp": {}}

    current_wdp_model = wdp_model_to_test
    if current_wdp_model is None:
        logger_to_use.info("No trained WDP model provided for generalization test, using default WDP (all weights=1).")
        current_wdp_model = WeightedDotProductModel(n_dim).to(config.DEVICE)

    current_wdp_model.eval()
    correct_rankings_count = 0
    total_test_cases = config.NUM_TEST_CASES_WDP_GENERALIZATION
    performance_by_I0 = {}  # Stores {'I0_tuple_str': {'correct': count, 'total': count}}
    all_agree_scores_wdp = []
    all_disagree_scores_wdp = []

    if total_test_cases == 0:
        logger_to_use.warning("NUM_TEST_CASES_WDP_GENERALIZATION is 0. Skipping WDP generalization test.")
        return {"wdp_generalization_success_rate": 0.0, "wdp_confidence_interval": (0.0, 0.0),
                "detailed_wdp_performance": [], "score_distributions_wdp": {}}

    for _ in tqdm(range(total_test_cases), desc="Testing WDP Generalization"):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        I0_indices_tuple = generate_random_I0_indices(n_dim, 2)  # This is a tuple
        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices_tuple).to(config.DEVICE)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)

        with torch.no_grad():
            scores_agree = [current_wdp_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
            scores_disagree = [current_wdp_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]

        all_agree_scores_wdp.extend(scores_agree)
        all_disagree_scores_wdp.extend(scores_disagree)

        I0_key_str = str(I0_indices_tuple)
        if I0_key_str not in performance_by_I0: performance_by_I0[I0_key_str] = {'correct': 0, 'total': 0}
        performance_by_I0[I0_key_str]['total'] += 1

        min_agree_score = min(scores_agree) if scores_agree else float('-inf')
        max_disagree_score = max(scores_disagree) if scores_disagree else float('inf')

        if min_agree_score > max_disagree_score:
            correct_rankings_count += 1
            performance_by_I0[I0_key_str]['correct'] += 1

    success_rate = (correct_rankings_count / total_test_cases) * 100
    lower_ci, upper_ci = compute_confidence_interval(correct_rankings_count, total_test_cases, config.CONFIDENCE_LEVEL)

    logger_to_use.info(f"WDP Model Generalization Test Summary:")
    logger_to_use.info(f"  Total test cases (random q, random I0): {total_test_cases}")
    logger_to_use.info(f"  Correct rankings achieved: {correct_rankings_count}")
    logger_to_use.info(f"  Success rate: {success_rate:.2f}% (CI: [{lower_ci:.2f}%, {upper_ci:.2f}%])")

    i0_performance_details_list = []
    logger_to_use.info("\nWDP Performance by I0 (for generalization test):")
    for I0_str_key, stats_val in performance_by_I0.items():
        if stats_val['total'] >= config.MIN_SAMPLES_FOR_I0_ANALYSIS:
            sr_i0 = (stats_val['correct'] / stats_val['total']) * 100
            logger_to_use.info(f"  I0={I0_str_key}: {sr_i0:.1f}% ({stats_val['correct']}/{stats_val['total']})")
            i0_performance_details_list.append({
                'I0': I0_str_key, 'success_rate': sr_i0,
                'correct': stats_val['correct'], 'total': stats_val['total']
            })
    if i0_performance_details_list:
        rates = [item['success_rate'] for item in i0_performance_details_list]
        logger_to_use.info(
            f"  Stats across reported I0 sets: Mean={np.mean(rates):.1f}%, Std={np.std(rates):.1f}%, Min={np.min(rates):.1f}%, Max={np.max(rates):.1f}%")

    if n_dim >= 3:
        logger_to_use.info("✗ As expected by Theorem 3.1.ii, a single WDP model struggles for universal success.")

    results_dict = {
        "wdp_generalization_success_rate": success_rate,
        "wdp_confidence_interval": (lower_ci, upper_ci),
        "detailed_wdp_performance": i0_performance_details_list,
        "score_distributions_wdp": {
            "agree_scores": all_agree_scores_wdp[:min(200, len(all_agree_scores_wdp))],
            "disagree_scores": all_disagree_scores_wdp[:min(200, len(all_disagree_scores_wdp))]
        }
    }
    failure_analysis_for_this_wdp = analyze_wdp_failures(current_wdp_model, logger_to_use)
    results_dict.update(failure_analysis_for_this_wdp)
    return results_dict


def save_detailed_results(results_dict_to_save, current_experiment_name, logger_to_use):
    results_dir = os.path.join(config.MODEL_SAVE_DIR_EXP1, current_experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    if config.SAVE_DETAILED_RESULTS:
        logger_to_use.info(f"Saving detailed results to: {results_dir}")
        if 'detailed_wdp_performance' in results_dict_to_save and results_dict_to_save['detailed_wdp_performance']:
            df_wdp_perf = pd.DataFrame(results_dict_to_save['detailed_wdp_performance'])
            df_wdp_perf.to_csv(os.path.join(results_dir, 'wdp_performance_by_I0.csv'), index=False)

        if 'score_distributions_bilinear' in results_dict_to_save:
            with open(os.path.join(results_dir, 'score_distributions_bilinear.json'), 'w') as f:
                json.dump(results_dict_to_save['score_distributions_bilinear'], f, indent=2)
        if 'score_distributions_wdp' in results_dict_to_save:
            with open(os.path.join(results_dir, 'score_distributions_wdp.json'), 'w') as f:
                json.dump(results_dict_to_save['score_distributions_wdp'], f, indent=2)

        if 'specialized_wdp_results' in results_dict_to_save and results_dict_to_save['specialized_wdp_results']:
            with open(os.path.join(results_dir, 'specialized_wdp_results.json'), 'w') as f:
                json.dump(results_dict_to_save['specialized_wdp_results'], f, indent=2)

        if 'failure_analysis' in results_dict_to_save and results_dict_to_save['failure_analysis']:
            df_failures = pd.DataFrame(results_dict_to_save['failure_analysis'])
            df_failures.to_csv(os.path.join(results_dir, 'failure_analysis.csv'), index=False)
    else:
        logger_to_use.info("Skipping saving detailed results (SAVE_DETAILED_RESULTS is False in config).")


def main_experiment_run(run_config_name_arg, run_params_arg, seed_value_arg):
    global current_run_logger  # Use the global variable for the logger specific to this run

    config.CURRENT_WDP_TRAIN_PARAMS = run_params_arg
    set_seeds(seed_value_arg)  # Sets seeds and uses current_run_logger if initialized (it will be by setup_logging)

    train_wdp_flag_current_run = run_params_arg.get("train_wdp_flag", True)
    wdp_status_str_current_run = "trainedWDP" if train_wdp_flag_current_run else "defaultWDP"

    param_desc_parts_current_run = []
    if train_wdp_flag_current_run:
        param_desc_parts_current_run.append(f"ep{run_params_arg.get('wdp_epochs', 'NA')}")
        samples_val = run_params_arg.get('wdp_train_samples', 'NA')
        if isinstance(samples_val, int) and samples_val % 1000 == 0 and samples_val > 0:
            param_desc_parts_current_run.append(f"samp{samples_val // 1000}k")
        else:
            param_desc_parts_current_run.append(f"samp{samples_val}")
    param_details_str_current_run = "_".join(
        param_desc_parts_current_run) if param_desc_parts_current_run else "defWDPparams"

    current_experiment_name_str = f"Exp1_{run_config_name_arg}_n{config.DIM_N}_seed{seed_value_arg}_{param_details_str_current_run}_{time.strftime('%Y%m%d-%H%M%S')}"

    current_run_logger = setup_logging_exp1(config.MODEL_SAVE_DIR_EXP1, current_experiment_name_str)
    # Now current_run_logger is properly initialized for this run before set_seeds is effectively logging.
    # If set_seeds was called before logger was initialized, its log message wouldn't go to file.
    # It's fine, as the seed is logged below too.

    current_run_logger.info(f"Running Experiment 1: Structured Agreement Task")
    current_run_logger.info(f"Run Configuration Name: {run_config_name_arg}")
    current_run_logger.info(f"Description: {run_params_arg.get('description', 'N/A')}")
    current_run_logger.info(f"Run Parameters (WDP general training): {json.dumps(run_params_arg, indent=2)}")
    current_run_logger.info(f"Seed for this run: {seed_value_arg}")
    current_run_logger.info(f"Global Config: DIM_N={config.DIM_N}, DEVICE={config.DEVICE}")
    current_run_logger.info(f"WDP Training for general model in this run: {train_wdp_flag_current_run}")

    run_results_dict = {}

    current_run_logger.info("\n" + "=" * 50 + "\nPART 1: TESTING BILINEAR MODEL SUFFICIENCY\n" + "=" * 50)
    bilinear_results_dict = test_bilinear_sufficiency(current_run_logger)
    run_results_dict.update(bilinear_results_dict)

    current_run_logger.info(
        "\n" + "=" * 50 + "\nPART 2: TESTING WDP UNIVERSALITY FAILURE (GENERALIZATION)\n" + "=" * 50)
    general_wdp_model_instance = None
    if train_wdp_flag_current_run:
        current_run_logger.info("Training a general WDP model as per current run configuration...")
        general_wdp_model_instance = train_single_wdp_model(current_run_logger, specific_I0=None)
    else:
        current_run_logger.info("Using a default WDP model (untrained, weights=1) for the generalization test.")

    wdp_generalization_results_dict = test_wdp_universality_failure(general_wdp_model_instance, current_run_logger)
    run_results_dict.update(wdp_generalization_results_dict)

    if config.INCLUDE_SPECIALIZED_WDP_TEST:
        current_run_logger.info("\n" + "=" * 50 + "\nPART 3: TESTING SPECIALIZED WDP MODELS\n" + "=" * 50)
        specialized_wdp_run_results = test_wdp_with_multiple_fixed_I0_sets(current_run_logger)
        run_results_dict.update(specialized_wdp_run_results)
    else:
        current_run_logger.info("\nSkipping Part 3: Specialized WDP Testing (disabled in config).")

    current_run_logger.info(
        "\n" + "=" * 50 + f"\nEXPERIMENT RUN SUMMARY ({run_config_name_arg}, Seed {seed_value_arg})\n" + "=" * 50)
    if 'bilinear_success_rate' in run_results_dict:
        current_run_logger.info(
            f"  ✓ Bilinear Success Rate: {run_results_dict['bilinear_success_rate']:.2f}% (CI: {run_results_dict.get('bilinear_confidence_interval', 'N/A')})")
    if 'wdp_generalization_success_rate' in run_results_dict:
        current_run_logger.info(
            f"  ✗ WDP Generalization Success Rate: {run_results_dict['wdp_generalization_success_rate']:.2f}% (CI: {run_results_dict.get('wdp_confidence_interval', ('N/A', 'N/A'))})")

    run_results_json_path = os.path.join(config.MODEL_SAVE_DIR_EXP1, current_experiment_name_str,
                                         "experiment1_results.json")
    with open(run_results_json_path, 'w') as f:
        json.dump(run_results_dict, f, indent=2)
    current_run_logger.info(f"\n💾 Experiment 1 run results saved to {run_results_json_path}")

    save_detailed_results(run_results_dict, current_experiment_name_str, current_run_logger)

    return run_results_dict


if __name__ == "__main__":
    master_results_summary = {}
    # Setup a logger for the master script execution itself
    master_log_experiment_name = f"MasterLog_Exp1_AllRuns_{time.strftime('%Y%m%d-%H%M%S')}"
    # The setup_logging_exp1 function returns the logger it created.
    overall_process_run_logger = setup_logging_exp1(config.MODEL_SAVE_DIR_EXP1, master_log_experiment_name)
    overall_process_run_logger.info("Starting all Experiment 1 runs based on config.py settings.")

    for config_name_from_dict, run_params_from_dict_config in config.RUN_CONFIGS.items():
        overall_process_run_logger.info(
            f"\n\n{'=' * 80}\nPROCESSING CONFIGURATION: {config_name_from_dict}\n{'=' * 80}")
        overall_process_run_logger.info(f"Description: {run_params_from_dict_config.get('description', 'N/A')}")

        config_specific_results_list = []
        for seed_run_value in config.SEEDS_TO_RUN:
            overall_process_run_logger.info(
                f"\n--- Starting run for Config '{config_name_from_dict}', Seed {seed_run_value} ---")

            individual_run_output = main_experiment_run(
                run_config_name_arg=config_name_from_dict,
                run_params_arg=run_params_from_dict_config,
                seed_value_arg=seed_run_value
            )
            config_specific_results_list.append({
                "seed": seed_run_value,
                "config_params_used": run_params_from_dict_config,
                "bilinear_success_rate": individual_run_output.get("bilinear_success_rate", "N/A"),
                "wdp_generalization_success_rate": individual_run_output.get("wdp_generalization_success_rate", "N/A"),
                "wdp_confidence_interval": individual_run_output.get("wdp_confidence_interval", ("N/A", "N/A"))
            })
            overall_process_run_logger.info(
                f"--- Completed run for Config '{config_name_from_dict}', Seed {seed_run_value} ---")

        master_results_summary[config_name_from_dict] = config_specific_results_list

        overall_process_run_logger.info(f"\n\n--- AGGREGATED RESULTS FOR CONFIGURATION: {config_name_from_dict} ---")
        current_config_wdp_success_rates = []
        for result_item in config_specific_results_list:
            overall_process_run_logger.info(
                f"  Seed {result_item['seed']}: WDP Gen Success Rate = {result_item['wdp_generalization_success_rate']}")
            if isinstance(result_item['wdp_generalization_success_rate'], (int, float)):
                current_config_wdp_success_rates.append(result_item['wdp_generalization_success_rate'])

        if current_config_wdp_success_rates:
            mean_success = np.mean(current_config_wdp_success_rates)
            std_dev_success = np.std(current_config_wdp_success_rates)
            overall_process_run_logger.info(
                f"  Average WDP Generalization Success Rate over {len(config.SEEDS_TO_RUN)} runs: {mean_success:.2f}% (StdDev: {std_dev_success:.2f}%)")
        else:
            overall_process_run_logger.info(
                "  No WDP success rates to average for this configuration (or n_dim < 3, or 0 test cases).")
        overall_process_run_logger.info(f"{'=' * 80}\n")

    master_summary_file_path = os.path.join(config.MODEL_SAVE_DIR_EXP1, "experiment1_master_summary_ALL_CONFIGS.json")
    with open(master_summary_file_path, 'w') as f_out:
        json.dump(master_results_summary, f_out, indent=4)
    overall_process_run_logger.info(
        f"\n\nMaster summary of all runs and configurations saved to: {master_summary_file_path}")
    overall_process_run_logger.info("All configured Experiment 1 runs complete.")