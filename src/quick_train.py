# quick_train.py
"""
Quick training script for testing the pipeline with a small dataset
"""
import torch
import os
from tqdm import tqdm

import config
from models import get_model
from data_loader import load_embeddings_and_mappings, load_dev_data_for_eval, MSMARCOTriplesDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def quick_train_test():
    """Quick training test with minimal data"""
    print("Quick Training Test")
    print("=" * 50)

    # Override config for quick test
    config.BATCH_SIZE = 16
    config.NUM_EPOCHS = 1
    config.LOG_INTERVAL = 10

    # Load data
    print("Loading embeddings...")
    query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx = load_embeddings_and_mappings()

    # Create small dataset (limit to 100 triples)
    print("Creating small dataset...")
    train_dataset = MSMARCOTriplesDataset(
        config.TRAIN_TRIPLES_PATH,
        qid_to_idx,
        pid_to_idx,
        query_embeddings,
        passage_embeddings,
        limit_size=100  # Only 100 triples for quick test
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Test dot product model (no training needed)
    print("\nTesting Dot Product Model...")
    model_config = config.MODEL_CONFIGS["dot_product"]
    model = get_model("dot_product", model_config).to(config.DEVICE)

    # Test a batch
    print("Testing forward pass...")
    for batch in train_dataloader:
        q_embeds, pos_p_embeds, neg_p_embeds = batch
        q_embeds = q_embeds.to(config.DEVICE)
        pos_p_embeds = pos_p_embeds.to(config.DEVICE)
        neg_p_embeds = neg_p_embeds.to(config.DEVICE)

        with torch.no_grad():
            scores_pos = model(q_embeds, pos_p_embeds)
            scores_neg = model(q_embeds, neg_p_embeds)

        print(f"Positive scores shape: {scores_pos.shape}")
        print(f"Negative scores shape: {scores_neg.shape}")
        print(f"Sample positive score: {scores_pos[0].item():.4f}")
        print(f"Sample negative score: {scores_neg[0].item():.4f}")
        break

    # Test weighted dot product model (quick training)
    print("\nTesting Weighted Dot Product Model Training...")
    model_config = config.MODEL_CONFIGS["weighted_dot_product"]
    model = get_model("weighted_dot_product", model_config).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MarginRankingLoss(margin=config.MARGIN)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train for a few batches
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc="Quick Training")
    for i, (q_embeds, pos_p_embeds, neg_p_embeds) in enumerate(progress_bar):
        q_embeds = q_embeds.to(config.DEVICE)
        pos_p_embeds = pos_p_embeds.to(config.DEVICE)
        neg_p_embeds = neg_p_embeds.to(config.DEVICE)

        optimizer.zero_grad()

        scores_pos = model(q_embeds, pos_p_embeds)
        scores_neg = model(q_embeds, neg_p_embeds)

        targets = torch.ones_like(scores_pos).to(config.DEVICE)
        loss = loss_fn(scores_pos, scores_neg, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (i + 1)})

        if i >= 5:  # Only train for 5 batches
            break

    print(f"Final average loss: {total_loss / min(6, len(train_dataloader)):.4f}")

    # Test evaluation setup
    print("\nTesting evaluation setup...")
    dev_query_to_candidates = load_dev_data_for_eval(
        config.DEV_QUERIES_PATH,
        config.DEV_CANDIDATES_PATH,
        qid_to_idx,
        pid_to_idx
    )

    if dev_query_to_candidates:
        # Test with one query
        sample_qid = list(dev_query_to_candidates.keys())[0]
        sample_candidates = dev_query_to_candidates[sample_qid][:5]  # Only 5 candidates

        print(f"Sample query: {sample_qid}")
        print(f"Number of candidates: {len(sample_candidates)}")

        # Quick scoring test
        q_embed_idx = qid_to_idx[sample_qid]
        q_embed = torch.tensor(query_embeddings[q_embed_idx], dtype=torch.float).unsqueeze(0).to(config.DEVICE)

        candidate_embeds = []
        for pid in sample_candidates:
            p_embed_idx = pid_to_idx[pid]
            candidate_embeds.append(passage_embeddings[p_embed_idx])

        candidate_embeds = torch.tensor(candidate_embeds, dtype=torch.float).to(config.DEVICE)
        q_embed_expanded = q_embed.expand(len(sample_candidates), -1)

        with torch.no_grad():
            scores = model(q_embed_expanded, candidate_embeds)
            sorted_indices = torch.argsort(scores, descending=True)

        print("Candidate scores:")
        for i, idx in enumerate(sorted_indices):
            print(f"  Rank {i + 1}: PID {sample_candidates[idx.item()]} (score: {scores[idx.item()].item():.4f})")

    print("\nQuick test completed successfully! ✓")


if __name__ == "__main__":
    try:
        quick_train_test()
    except Exception as e:
        print(f"\nError during quick test: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease check:")
        print("1. All required files are downloaded")
        print("2. Embeddings have been generated (run preprocess_embeddings.py)")
        print("3. Paths in config.py are correct")