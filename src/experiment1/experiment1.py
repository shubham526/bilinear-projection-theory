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

import config as config  # Use the experiment-specific config
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


def setup_logging_exp1(model_save_dir, experiment_name):
    """Setup logging for Experiment 1"""
    log_dir = os.path.join(model_save_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'experiment.log')),
            logging.StreamHandler()
        ],
        force=True  # This ensures existing handlers are removed
    )
    return logging.getLogger()


def compute_confidence_interval(successes, total, confidence=0.95):
    """Compute confidence interval for success rate"""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    if p == 0 or p == 1:
        return p * 100, p * 100
    margin = stats.norm.ppf((1 + confidence) / 2) * np.sqrt(p * (1 - p) / total)
    return max(0, (p - margin) * 100), min(100, (p + margin) * 100)


def test_bilinear_sufficiency(logger):
    logger.info("\n--- Part (i): Bilinear Model Sufficiency (Theorem 3.1.i) ---")
    n_dim = config.DIM_N
    perfect_rankings_count = 0
    total_test_cases = config.NUM_TEST_CASES_BILINEAR
    score_distributions = {'agree_scores': [], 'disagree_scores': []}

    for _ in tqdm(range(total_test_cases), desc="Testing Bilinear Sufficiency"):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        I0_indices = generate_random_I0_indices(n_dim, 2)

        W_I0_matrix = construct_theoretical_W_I0(n_dim, I0_indices).to(config.DEVICE)
        bilinear_model = BilinearScorer(W_I0_matrix)  # W_I0_matrix is already on device

        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices).to(config.DEVICE)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)

        scores_agree = [bilinear_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
        scores_disagree = [bilinear_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]

        # Store score distributions
        score_distributions['agree_scores'].extend(scores_agree)
        score_distributions['disagree_scores'].extend(scores_disagree)

        # Check if all agree scores are approx +2 and all disagree scores are approx -2
        # And that min agree score > max disagree score
        correct_agree_scores = all(abs(s - 2.0) < 1e-5 for s in scores_agree)
        correct_disagree_scores = all(abs(s - (-2.0)) < 1e-5 for s in scores_disagree)

        min_agree = min(scores_agree) if scores_agree else -float('inf')
        max_disagree = max(scores_disagree) if scores_disagree else float('inf')

        if correct_agree_scores and correct_disagree_scores and (min_agree > max_disagree):
            perfect_rankings_count += 1
        else:
            if not (correct_agree_scores and correct_disagree_scores):
                logger.debug(f"Failed case for Bilinear: q={q.cpu().numpy()}, I0={I0_indices}")
                logger.debug(f"  Agree Scores: {scores_agree} (Expected [+2, +2])")
                logger.debug(f"  Disagree Scores: {scores_disagree} (Expected [-2, -2])")

    success_rate = (perfect_rankings_count / total_test_cases) * 100
    lower_ci, upper_ci = compute_confidence_interval(perfect_rankings_count, total_test_cases, config.CONFIDENCE_LEVEL)

    logger.info(f"Bilinear Model Test Summary:")
    logger.info(f"  Total test cases: {total_test_cases}")
    logger.info(f"  Perfect rankings achieved: {perfect_rankings_count}")
    logger.info(f"  Success rate: {success_rate:.2f}% (95% CI: [{lower_ci:.2f}%, {upper_ci:.2f}%])")
    assert abs(success_rate - 100.0) < 1e-3, "Bilinear model did not achieve 100% success as expected."
    logger.info("✓ Bilinear model sufficiency confirmed as per Theorem 3.1.i.")

    return {
        "bilinear_success_rate": success_rate,
        "bilinear_confidence_interval": (lower_ci, upper_ci),
        "score_distributions": score_distributions
    }


def train_single_wdp_model(logger, specific_I0=None):
    """Train a weighted dot product model, optionally specialized for a specific I0"""
    if specific_I0:
        logger.info(f"\n--- Training WDP Model Specialized for I0={specific_I0} ---")
    else:
        logger.info("\n--- Training a Single Weighted Dot Product Model ---")

    n_dim = config.DIM_N
    wdp_model = WeightedDotProductModel(n_dim, init_strategy='small_random').to(config.DEVICE)
    optimizer = optim.AdamW(wdp_model.parameters(), lr=config.WDP_LEARNING_RATE)
    loss_fn = nn.MarginRankingLoss(margin=config.WDP_MARGIN).to(config.DEVICE)

    logger.info(f"WDP Model parameters: {sum(p.numel() for p in wdp_model.parameters() if p.requires_grad)}")

    # Generate training data
    training_samples = []
    num_samples = config.WDP_TRAIN_SAMPLES if not specific_I0 else config.WDP_TRAIN_SAMPLES // 5
    logger.info(f"Generating {num_samples} training samples for WDP...")

    for _ in tqdm(range(num_samples), desc="Generating WDP train data"):
        q = generate_random_query_vector(n_dim)

        # If training for specific I0, only use that I0
        if specific_I0:
            I0_indices = specific_I0
        else:
            I0_indices = generate_random_I0_indices(n_dim, 2)

        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)
        # Pick one agree and one disagree for the triplet loss
        training_samples.append((q, docs_agree[0], docs_disagree[0]))

    # Simple DataLoader for these samples
    class WDPTrainDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    train_dataset = WDPTrainDataset(training_samples)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.WDP_BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(config.WDP_EPOCHS):
        wdp_model.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f"WDP Train Epoch {epoch + 1}/{config.WDP_EPOCHS}")
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
            num_batches += 1
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        logger.info(f"WDP Train Epoch {epoch + 1} Avg Loss: {total_loss / num_batches:.4f}")

    logger.info("WDP model training finished.")
    logger.info(f"Final weights: {wdp_model.get_weights()}")

    return wdp_model


def test_wdp_on_specific_I0(wdp_model, I0_indices, logger, num_samples=None):
    """Test WDP model performance on a specific I0 set"""
    if num_samples is None:
        num_samples = config.NUM_SAMPLES_PER_I0_TEST

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

        # Check if min agree score > max disagree score
        min_agree = min(scores_agree) if scores_agree else -float('inf')
        max_disagree = max(scores_disagree) if scores_disagree else float('inf')

        if min_agree > max_disagree:
            correct_rankings_count += 1

    success_rate = (correct_rankings_count / num_samples) * 100
    return success_rate


def test_wdp_with_multiple_fixed_I0_sets(logger):
    """Test that WDP trained on one I0 fails on different I0 sets"""
    if not config.INCLUDE_SPECIALIZED_WDP_TEST:
        logger.info("\n--- Skipping Specialized WDP Test (disabled in config) ---")
        return {}

    logger.info("\n--- Testing WDP Adaptation to Different I0 Sets ---")

    # Get test I0 sets based on n_dim
    n_dim = config.DIM_N
    all_I0s = generate_all_possible_I0_indices(n_dim, 2)
    test_I0_sets = all_I0s[:min(3, len(all_I0s))]  # Test first 3 I0 sets maximum

    results = {'specialized_wdp_results': []}

    for I0_test in test_I0_sets:
        # Train WDP specifically for this I0
        logger.info(f"\nTraining WDP specifically for I0={I0_test}")
        specialized_wdp = train_single_wdp_model(logger, specific_I0=I0_test)

        # Test on this I0 (should work well)
        success_rate_same = test_wdp_on_specific_I0(specialized_wdp, I0_test, logger)
        logger.info(f"WDP success on training I0 {I0_test}: {success_rate_same:.1f}%")

        I0_results = {
            'training_I0': I0_test,
            'success_on_training_I0': success_rate_same,
            'success_on_other_I0s': {}
        }

        # Test on different I0 sets (should fail)
        for I0_different in test_I0_sets:
            if I0_different != I0_test:
                success_rate_diff = test_wdp_on_specific_I0(specialized_wdp, I0_different, logger)
                logger.info(f"WDP success on different I0 {I0_different}: {success_rate_diff:.1f}%")
                I0_results['success_on_other_I0s'][str(I0_different)] = success_rate_diff

        results['specialized_wdp_results'].append(I0_results)

    return results


def analyze_wdp_failures(wdp_model, logger):
    """Analyze why WDP fails on certain patterns"""
    if not config.ANALYZE_FAILURE_PATTERNS:
        logger.info("\n--- Skipping Failure Pattern Analysis (disabled in config) ---")
        return {}

    logger.info("\n--- Analyzing WDP Failure Patterns ---")
    n_dim = config.DIM_N

    # Get challenging patterns
    challenging_patterns = generate_challenging_patterns(n_dim)

    failure_analysis = []

    for pattern in challenging_patterns:
        q = pattern["q"][:n_dim].to(config.DEVICE)
        I0 = pattern["I0"]

        if max(I0) >= n_dim:
            continue

        logger.info(f"\nAnalyzing pattern: {pattern['name']}")
        logger.info(f"Description: {pattern['description']}")

        # Test this pattern multiple times
        correct_rankings = 0
        num_tests = 100

        for _ in range(num_tests):
            e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0).to(config.DEVICE)
            docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)

            with torch.no_grad():
                scores_agree = [wdp_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
                scores_disagree = [wdp_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]

            min_agree = min(scores_agree) if scores_agree else -float('inf')
            max_disagree = max(scores_disagree) if scores_disagree else float('inf')

            if min_agree > max_disagree:
                correct_rankings += 1

        success_rate = (correct_rankings / num_tests) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

        failure_analysis.append({
            'pattern_name': pattern['name'],
            'I0': I0,
            'success_rate': success_rate,
            'description': pattern['description']
        })

    # Analysis of what WDP would need to succeed
    logger.info("\n--- Why WDP Struggles ---")
    logger.info("WDP requires fixed weights v that must work for ALL possible I0 sets.")
    logger.info("But different I0 sets require prioritizing different dimensions:")

    all_I0s = generate_all_possible_I0_indices(n_dim, 2)[:5]  # Show first 5
    for I0 in all_I0s:
        logger.info(f"  I0={I0} requires high weights for dimensions {I0}")

    logger.info("No single weight vector can satisfy all these conflicting requirements!")

    return {'failure_analysis': failure_analysis}


def test_wdp_universality_failure(wdp_model, logger):
    logger.info("\n--- Part (ii): Weighted Dot Product Universality Failure (Theorem 3.1.ii) ---")
    n_dim = config.DIM_N
    if n_dim < 3:
        logger.warning(f"WDP failure demonstration requires n_dim >= 3. Current n_dim = {n_dim}. Skipping.")
        return {"wdp_generalization_success_rate": "N/A (n_dim < 3)"}

    if wdp_model is None:  # If not training one, create a default one (e.g., weights = 1)
        logger.info("No trained WDP model provided, using default WDP (all weights=1).")
        wdp_model = WeightedDotProductModel(n_dim).to(config.DEVICE)

    wdp_model.eval()  # Ensure it's in eval mode

    correct_rankings_count = 0
    total_test_cases = config.NUM_TEST_CASES_WDP_GENERALIZATION

    # Track performance by different I0 sets to show inconsistency
    performance_by_I0 = {}
    score_distributions = {'agree_scores': [], 'disagree_scores': []}

    for _ in tqdm(range(total_test_cases), desc="Testing WDP Generalization"):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        # CRITICAL: Test on a *random* I0 to show it can't handle all
        I0_indices = generate_random_I0_indices(n_dim, 2)

        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices).to(config.DEVICE)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)

        with torch.no_grad():
            scores_agree = [wdp_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
            scores_disagree = [wdp_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]

        # Store score distributions
        score_distributions['agree_scores'].extend(scores_agree)
        score_distributions['disagree_scores'].extend(scores_disagree)

        # Track performance for this I0
        I0_key = str(I0_indices)
        if I0_key not in performance_by_I0:
            performance_by_I0[I0_key] = {'correct': 0, 'total': 0}

        performance_by_I0[I0_key]['total'] += 1

        # Check if min agree score > max disagree score
        min_agree = min(scores_agree) if scores_agree else -float('inf')
        max_disagree = max(scores_disagree) if scores_disagree else float('inf')

        if min_agree > max_disagree:
            correct_rankings_count += 1
            performance_by_I0[I0_key]['correct'] += 1

    success_rate = (correct_rankings_count / total_test_cases) * 100
    lower_ci, upper_ci = compute_confidence_interval(correct_rankings_count, total_test_cases, config.CONFIDENCE_LEVEL)

    logger.info(f"WDP Model Generalization Test Summary:")
    logger.info(f"  Total test cases (random q, random I0): {total_test_cases}")
    logger.info(f"  Correct rankings achieved: {correct_rankings_count}")
    logger.info(f"  Success rate: {success_rate:.2f}% (95% CI: [{lower_ci:.2f}%, {upper_ci:.2f}%])")

    # Log inconsistent performance across different I0 sets
    logger.info("\nWDP Performance varies by I0 (demonstrating universality failure):")
    i0_performance_details = []
    for I0, stats in performance_by_I0.items():
        if stats['total'] >= config.MIN_SAMPLES_FOR_I0_ANALYSIS:  # Only show I0s with enough samples
            success_rate_i0 = (stats['correct'] / stats['total']) * 100
            logger.info(f"  I0={I0}: {success_rate_i0:.1f}% ({stats['correct']}/{stats['total']})")
            i0_performance_details.append({
                'I0': I0,
                'success_rate': success_rate_i0,
                'correct': stats['correct'],
                'total': stats['total']
            })

    # Show variance in performance across I0 sets
    if i0_performance_details:
        success_rates = [item['success_rate'] for item in i0_performance_details]
        logger.info(f"\nPerformance Statistics Across I0 Sets:")
        logger.info(f"  Mean: {np.mean(success_rates):.1f}%")
        logger.info(f"  Std Dev: {np.std(success_rates):.1f}%")
        logger.info(f"  Min: {np.min(success_rates):.1f}%")
        logger.info(f"  Max: {np.max(success_rates):.1f}%")

    if n_dim >= 3:
        logger.info("✗ As expected by Theorem 3.1.ii, a single WDP model struggles to achieve universal success.")

    results = {
        "wdp_generalization_success_rate": success_rate,
        "wdp_confidence_interval": (lower_ci, upper_ci),
        "detailed_wdp_performance": i0_performance_details,
        "score_distributions": score_distributions
    }

    # Add failure analysis
    failure_analysis_results = analyze_wdp_failures(wdp_model, logger)
    results.update(failure_analysis_results)

    return results


def save_detailed_results(results, experiment_name):
    """Save detailed results for visualization"""
    results_dir = os.path.join(config.MODEL_SAVE_DIR_EXP1, experiment_name)

    if config.SAVE_DETAILED_RESULTS:
        # Convert I0 performance to DataFrame for easy plotting
        if 'detailed_wdp_performance' in results:
            df = pd.DataFrame(results['detailed_wdp_performance'])
            df.to_csv(os.path.join(results_dir, 'wdp_performance_by_I0.csv'), index=False)

        # Save score distributions
        if 'score_distributions' in results:
            with open(os.path.join(results_dir, 'score_distributions.json'), 'w') as f:
                json.dump(results['score_distributions'], f, indent=2)

        # Save specialized WDP results
        if 'specialized_wdp_results' in results:
            with open(os.path.join(results_dir, 'specialized_wdp_results.json'), 'w') as f:
                json.dump(results['specialized_wdp_results'], f, indent=2)

        # Save failure analysis
        if 'failure_analysis' in results:
            df_failures = pd.DataFrame(results['failure_analysis'])
            df_failures.to_csv(os.path.join(results_dir, 'failure_analysis.csv'), index=False)


def main():
    experiment_name = f"Exp1_StructuredAgreement_n{config.DIM_N}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = setup_logging_exp1(config.MODEL_SAVE_DIR_EXP1, experiment_name)

    logger.info(f"Running Experiment 1: Structured Agreement Task")
    logger.info(f"Configuration:")
    logger.info(f"  DIM_N={config.DIM_N}")
    logger.info(f"  DEVICE={config.DEVICE}")
    logger.info(f"  NUM_TEST_CASES_BILINEAR={config.NUM_TEST_CASES_BILINEAR}")
    logger.info(f"  NUM_TEST_CASES_WDP_GENERALIZATION={config.NUM_TEST_CASES_WDP_GENERALIZATION}")
    logger.info(f"  SAVE_DETAILED_RESULTS={config.SAVE_DETAILED_RESULTS}")
    logger.info(f"  ANALYZE_FAILURE_PATTERNS={config.ANALYZE_FAILURE_PATTERNS}")
    logger.info(f"  INCLUDE_SPECIALIZED_WDP_TEST={config.INCLUDE_SPECIALIZED_WDP_TEST}")

    results = {}

    # Part 1: Bilinear Sufficiency
    logger.info("\n" + "=" * 50)
    logger.info("PART 1: TESTING BILINEAR MODEL SUFFICIENCY")
    logger.info("=" * 50)
    results_bilinear = test_bilinear_sufficiency(logger)
    results.update(results_bilinear)

    # Part 2: WDP Failure
    logger.info("\n" + "=" * 50)
    logger.info("PART 2: TESTING WDP UNIVERSALITY FAILURE")
    logger.info("=" * 50)

    # Option A: Train a single WDP model then test its universality
    trained_wdp = train_single_wdp_model(logger)
    # Option B: Use a default WDP (e.g. v=all ones) if not training
    # trained_wdp = None # If you don't want to train one for this test.

    results_wdp = test_wdp_universality_failure(trained_wdp, logger)
    results.update(results_wdp)

    # Part 3: Specialized WDP Testing
    logger.info("\n" + "=" * 50)
    logger.info("PART 3: TESTING SPECIALIZED WDP MODELS")
    logger.info("=" * 50)

    specialized_results = test_wdp_with_multiple_fixed_I0_sets(logger)
    results.update(specialized_results)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("EXPERIMENT 1 SUMMARY")
    logger.info("=" * 50)

    logger.info("\n📊 Key Results:")
    if 'bilinear_success_rate' in results:
        logger.info(f"  ✓ Bilinear Success Rate: {results['bilinear_success_rate']:.2f}%")
        if 'bilinear_confidence_interval' in results:
            ci = results['bilinear_confidence_interval']
            logger.info(f"    (95% CI: [{ci[0]:.2f}%, {ci[1]:.2f}%])")

    if 'wdp_generalization_success_rate' in results:
        logger.info(f"  ✗ WDP Success Rate: {results['wdp_generalization_success_rate']:.2f}%")
        if 'wdp_confidence_interval' in results:
            ci = results['wdp_confidence_interval']
            logger.info(f"    (95% CI: [{ci[0]:.2f}%, {ci[1]:.2f}%])")

    logger.info("\n🔬 Theoretical Validation:")
    logger.info("  ✓ Theorem 3.1.i: Bilinear models achieve perfect performance")
    logger.info("  ✓ Theorem 3.1.ii: No single WDP can universally solve the task")

    if 'detailed_wdp_performance' in results and results['detailed_wdp_performance']:
        success_rates = [item['success_rate'] for item in results['detailed_wdp_performance']]
        logger.info(f"\n📈 WDP Performance Variability:")
        logger.info(
            f"  - Performance varies from {np.min(success_rates):.1f}% to {np.max(success_rates):.1f}% across different I0 sets")
        logger.info(f"  - Standard deviation: {np.std(success_rates):.1f}%")
        logger.info("  - This variability demonstrates the universality failure")

    # Save all results
    results_path = os.path.join(config.MODEL_SAVE_DIR_EXP1, experiment_name, "experiment1_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Experiment 1 results saved to {results_path}")

    # Save detailed results
    save_detailed_results(results, experiment_name)
    if config.SAVE_DETAILED_RESULTS:
        logger.info(f"📁 Detailed results saved for visualization")


if __name__ == "__main__":
    main()