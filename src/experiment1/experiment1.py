# experiment1.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import json
import time
import logging

import config as config  # Use the experiment-specific config
from synthetic_data_gen import (
    generate_random_query_vector,
    generate_random_I0_indices,
    generate_indicator_e_I_hypercube,
    generate_structured_agreement_docs
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
        ]
    )
    return logging.getLogger()


def test_bilinear_sufficiency(logger):
    logger.info("\n--- Part (i): Bilinear Model Sufficiency (Theorem 3.1.i) ---")
    n_dim = config.DIM_N
    perfect_rankings_count = 0
    total_test_cases = config.NUM_TEST_CASES_BILINEAR

    for _ in tqdm(range(total_test_cases), desc="Testing Bilinear Sufficiency"):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        I0_indices = generate_random_I0_indices(n_dim, 2)

        W_I0_matrix = construct_theoretical_W_I0(n_dim, I0_indices).to(config.DEVICE)
        bilinear_model = BilinearScorer(W_I0_matrix)  # W_I0_matrix is already on device

        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices).to(config.DEVICE)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)

        scores_agree = [bilinear_model(q, d_a.to(config.DEVICE)).item() for d_a in docs_agree]
        scores_disagree = [bilinear_model(q, d_d.to(config.DEVICE)).item() for d_d in docs_disagree]

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
    logger.info(f"Bilinear Model Test Summary:")
    logger.info(f"  Total test cases: {total_test_cases}")
    logger.info(f"  Perfect rankings achieved: {perfect_rankings_count}")
    logger.info(f"  Success rate: {success_rate:.2f}%")
    assert abs(success_rate - 100.0) < 1e-3, "Bilinear model did not achieve 100% success as expected."
    logger.info("Bilinear model sufficiency confirmed as per Theorem 3.1.i.")
    return {"bilinear_success_rate": success_rate}


def train_single_wdp_model(logger):
    logger.info("\n--- Training a Single Weighted Dot Product Model ---")
    n_dim = config.DIM_N
    wdp_model = WeightedDotProductModel(n_dim).to(config.DEVICE)
    optimizer = optim.AdamW(wdp_model.parameters(), lr=config.WDP_LEARNING_RATE)
    loss_fn = nn.MarginRankingLoss(margin=config.WDP_MARGIN).to(config.DEVICE)

    logger.info(f"WDP Model parameters: {sum(p.numel() for p in wdp_model.parameters() if p.requires_grad)}")

    # Generate training data: (q, I0_indices, d_agree, d_disagree)
    # For simplicity, we'll generate data on the fly for each batch,
    # or create a small dataset. Let's create a list of samples.

    training_samples = []
    logger.info(f"Generating {config.WDP_TRAIN_SAMPLES} training samples for WDP...")
    for _ in tqdm(range(config.WDP_TRAIN_SAMPLES), desc="Generating WDP train data"):
        q = generate_random_query_vector(n_dim)
        I0_indices = generate_random_I0_indices(n_dim, 2)
        e_I0_hyper = generate_indicator_e_I_hypercube(n_dim, I0_indices)
        docs_agree, docs_disagree = generate_structured_agreement_docs(q, e_I0_hyper)
        # Pick one agree and one disagree for the triplet loss
        training_samples.append((q, docs_agree[0], docs_disagree[0]))  # Or randomly pick

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
    return wdp_model


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

    for _ in tqdm(range(total_test_cases), desc="Testing WDP Generalization"):
        q = generate_random_query_vector(n_dim).to(config.DEVICE)
        # CRITICAL: Test on a *different* I0 than what it might have predominantly seen if trained
        # or just random I0 to show it can't handle all.
        I0_indices = generate_random_I0_indices(n_dim, 2)

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
        else:
            # Optional: Log a few failure cases
            # logger.debug(f"WDP Failed: q={q.cpu().numpy()}, I0={I0_indices}, Agree: {scores_agree}, Disagree: {scores_disagree}")
            pass

    success_rate = (correct_rankings_count / total_test_cases) * 100
    logger.info(f"Trained WDP Model Generalization Test Summary:")
    logger.info(f"  Total test cases (random q, random I0): {total_test_cases}")
    logger.info(f"  Correct rankings achieved: {correct_rankings_count}")
    logger.info(f"  Success rate: {success_rate:.2f}%")

    if n_dim >= 3:
        logger.info("As expected by Theorem 3.1.ii, a single WDP model struggles to achieve universal success.")

    return {"wdp_generalization_success_rate": success_rate}


def main():
    experiment_name = f"Exp1_StructuredAgreement_n{config.DIM_N}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = setup_logging_exp1(config.MODEL_SAVE_DIR_EXP1, experiment_name)

    logger.info(f"Running Experiment 1: Structured Agreement Task")
    logger.info(f"Configuration: DIM_N={config.DIM_N}, DEVICE={config.DEVICE}")

    results = {}

    # Part 1: Bilinear Sufficiency
    results_bilinear = test_bilinear_sufficiency(logger)
    results.update(results_bilinear)

    # Part 2: WDP Failure
    # Option A: Train a single WDP model then test its universality
    trained_wdp = train_single_wdp_model(logger)
    # Option B: Use a default WDP (e.g. v=all ones) if not training
    # trained_wdp = None # If you don't want to train one for this test.

    results_wdp = test_wdp_universality_failure(trained_wdp, logger)
    results.update(results_wdp)

    logger.info("\n--- Experiment 1 Summary ---")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")

    results_path = os.path.join(config.MODEL_SAVE_DIR_EXP1, experiment_name, "experiment1_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment 1 results saved to {results_path}")


if __name__ == "__main__":
    main()
