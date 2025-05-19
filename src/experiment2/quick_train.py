#!/usr/bin/env python
# quick_train_ir_datasets.py
"""
Quick training script for testing the pipeline with a small dataset using ir_datasets
"""
import torch
from tqdm import tqdm
import ir_datasets
import numpy as np

import config
from models import get_model
import torch.optim as optim
import torch.nn as nn


def load_embeddings_and_mappings_ir():
    """
    Load embeddings and mappings using stored files from the config paths.
    """
    print("Loading embeddings and mappings from saved files...")

    # Check if files exist
    for file_path in [config.QUERY_EMBEDDINGS_PATH, config.PASSAGE_EMBEDDINGS_PATH,
                      config.QUERY_ID_TO_IDX_PATH, config.PASSAGE_ID_TO_IDX_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load embeddings
    query_embeddings = np.load(config.QUERY_EMBEDDINGS_PATH)
    passage_embeddings = np.load(config.PASSAGE_EMBEDDINGS_PATH)

    # Load mappings
    with open(config.QUERY_ID_TO_IDX_PATH, 'r') as f:
        qid_to_idx = json.load(f)

    with open(config.PASSAGE_ID_TO_IDX_PATH, 'r') as f:
        pid_to_idx = json.load(f)

    print(f"Loaded {len(query_embeddings)} query embeddings")
    print(f"Loaded {len(passage_embeddings)} passage embeddings")

    return query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx


class MSMARCOTriplesDatasetIR(torch.utils.data.Dataset):
    """
    Dataset for MS MARCO triples using precomputed embeddings with ir_datasets.
    """

    def __init__(self, qid_to_idx, pid_to_idx, query_embeddings, passage_embeddings, limit_size=None):
        """
        Initialize the dataset.

        Args:
            qid_to_idx: Mapping from query IDs to embedding indices
            pid_to_idx: Mapping from passage IDs to embedding indices
            query_embeddings: Query embeddings matrix (numpy array)
            passage_embeddings: Passage embeddings matrix (numpy array)
            limit_size: Optional size limit for testing (number of triples)
        """
        super(MSMARCOTriplesDatasetIR, self).__init__()
        self.query_embeddings = query_embeddings
        self.passage_embeddings = passage_embeddings
        self.qid_to_idx = qid_to_idx
        self.pid_to_idx = pid_to_idx

        # Load triples from ir_datasets
        print("Loading training triples from ir_datasets...")
        self.triples = []

        # Load train dataset with triples
        dataset = ir_datasets.load("msmarco-passage/train/triples-small")

        count = 0
        skipped = 0

        for docpair in tqdm(dataset.docpairs_iter(), desc="Loading training triples"):
            qid = docpair.query_id
            pos_pid = docpair.doc_id_a
            neg_pid = docpair.doc_id_b

            # Check if we have embeddings for all three IDs
            if qid in self.qid_to_idx and pos_pid in self.pid_to_idx and neg_pid in self.pid_to_idx:
                # Store indices to the embeddings
                self.triples.append((
                    self.qid_to_idx[qid],
                    self.pid_to_idx[pos_pid],
                    self.pid_to_idx[neg_pid]
                ))
                count += 1

                # Apply size limit if specified
                if limit_size is not None and count >= limit_size:
                    break
            else:
                skipped += 1

        print(f"Loaded {len(self.triples)} training triples")
        print(f"Skipped {skipped} triples due to missing embeddings")

    def __len__(self):
        """Return the number of triples in the dataset"""
        return len(self.triples)

    def __getitem__(self, idx):
        """Get a single training triple"""
        qidx, pos_pidx, neg_pidx = self.triples[idx]

        # Get embeddings for this triple
        q_embed = self.query_embeddings[qidx]
        pos_p_embed = self.passage_embeddings[pos_pidx]
        neg_p_embed = self.passage_embeddings[neg_pidx]

        # Convert to torch tensors
        q_embed = torch.FloatTensor(q_embed)
        pos_p_embed = torch.FloatTensor(pos_p_embed)
        neg_p_embed = torch.FloatTensor(neg_p_embed)

        return q_embed, pos_p_embed, neg_p_embed


def load_dev_data_for_eval_ir(qid_to_idx, pid_to_idx, limit_size=5):
    """
    Load a small sample of dev data for evaluation using ir_datasets.
    Only loads 'limit_size' candidates per query for quick testing.
    """
    print("Loading sample dev data using ir_datasets...")

    # Load dev dataset
    dev_dataset = ir_datasets.load("msmarco-passage/dev/small")

    # Get a list of query IDs
    query_ids = []
    for query in dev_dataset.queries_iter():
        if query.query_id in qid_to_idx:
            query_ids.append(query.query_id)
            if len(query_ids) >= 5:  # Just get a few queries for quick test
                break

    # Create dictionary mapping each query to its candidates
    dev_query_to_candidates = {}

    # Get the qrels to use as candidates
    for qid in query_ids:
        dev_query_to_candidates[qid] = []

    # Add qrels as candidates
    for qrel in dev_dataset.qrels_iter():
        qid = qrel.query_id
        if qid in query_ids and qrel.doc_id in pid_to_idx:
            dev_query_to_candidates[qid].append(qrel.doc_id)

    # If we don't have enough candidates, try to get more
    import os
    import json

    # Try to get some candidate docs from stored top1000 file if available
    # Or just use non-relevant docs as candidates
    all_doc_ids = list(pid_to_idx.keys())
    import random
    random.shuffle(all_doc_ids)

    # Ensure each query has at least 'limit_size' candidates
    for qid in query_ids:
        if len(dev_query_to_candidates[qid]) < limit_size:
            # Add random passages to reach the required number
            needed = limit_size - len(dev_query_to_candidates[qid])
            for pid in all_doc_ids:
                if pid not in dev_query_to_candidates[qid] and pid in pid_to_idx:
                    dev_query_to_candidates[qid].append(pid)
                    needed -= 1
                    if needed <= 0:
                        break

    print(f"Loaded candidates for {len(dev_query_to_candidates)} queries")
    for qid, candidates in dev_query_to_candidates.items():
        print(f"  Query {qid}: {len(candidates)} candidates")

    return dev_query_to_candidates


def quick_train_test():
    """Quick training test with minimal data using ir_datasets"""
    print("Quick Training Test (ir_datasets version)")
    print("=" * 50)

    # Import necessary modules
    import os
    import json

    # Override config for quick test
    config.BATCH_SIZE = 16
    config.NUM_EPOCHS = 1
    config.LOG_INTERVAL = 10

    # Load data
    print("Loading embeddings...")
    query_embeddings = np.load(config.QUERY_EMBEDDINGS_PATH)
    passage_embeddings = np.load(config.PASSAGE_EMBEDDINGS_PATH)

    with open(config.QUERY_ID_TO_IDX_PATH, 'r') as f:
        qid_to_idx = json.load(f)

    with open(config.PASSAGE_ID_TO_IDX_PATH, 'r') as f:
        pid_to_idx = json.load(f)

    # Create small dataset (limit to 100 triples)
    print("Creating small dataset...")
    train_dataset = MSMARCOTriplesDatasetIR(
        qid_to_idx,
        pid_to_idx,
        query_embeddings,
        passage_embeddings,
        limit_size=100  # Only 100 triples for quick test
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

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
    dev_query_to_candidates = load_dev_data_for_eval_ir(qid_to_idx, pid_to_idx)

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
    print("The ir_datasets version is working properly!")


if __name__ == "__main__":
    try:
        quick_train_test()
    except Exception as e:
        print(f"\nError during quick test: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease check:")
        print("1. Make sure ir_datasets is installed: pip install ir_datasets")
        print("2. Embeddings have been generated (run preprocess_embeddings_ir_datasets.py)")
        print("3. Paths in config.py for embedding outputs are correct")