#!/usr/bin/env python
# create_cv_triples.py - Generate and save training triples for CV training

import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict
from utils import load_folds

# Import config for output paths
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure config.py is in the same directory.")
    exit(1)


def load_embeddings_and_mappings(embeddings_prefix):
    """
    Load embeddings and mappings based on prefix.

    Args:
        embeddings_prefix: Prefix for embedding files (e.g., car or robust)

    Returns:
        Tuple of (query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx)
    """
    embeddings_prefix = embeddings_prefix.upper()

    # Get paths from config using the prefix
    query_embeddings_path = getattr(config, f"{embeddings_prefix}_QUERY_EMBEDDINGS_PATH")
    passage_embeddings_path = getattr(config, f"{embeddings_prefix}_PASSAGE_EMBEDDINGS_PATH")
    query_id_to_idx_path = getattr(config, f"{embeddings_prefix}_QUERY_ID_TO_IDX_PATH")
    passage_id_to_idx_path = getattr(config, f"{embeddings_prefix}_PASSAGE_ID_TO_IDX_PATH")

    print(f"Loading query embeddings from {query_embeddings_path}")
    query_embeddings = np.load(query_embeddings_path)

    print(f"Loading passage embeddings from {passage_embeddings_path}")
    passage_embeddings = np.load(passage_embeddings_path)

    print(f"Loading query ID to index mapping from {query_id_to_idx_path}")
    with open(query_id_to_idx_path, 'r') as f:
        qid_to_idx = json.load(f)

    print(f"Loading passage ID to index mapping from {passage_id_to_idx_path}")
    with open(passage_id_to_idx_path, 'r') as f:
        pid_to_idx = json.load(f)

    return query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx



def load_qrels(qrels_file, train_query_ids=None):
    """
    Load qrels file to get positive passages for each query.

    Args:
        qrels_file: Path to qrels file
        train_query_ids: Optional list of query IDs to filter by

    Returns:
        dict: Mapping from query IDs to lists of positive passage IDs
    """
    query_to_positive_pids = defaultdict(list)

    print(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r') as f:
        for line in tqdm(f, desc="Loading qrels"):
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                pid = parts[2]
                rel = int(parts[3])

                # Filter by train_query_ids if provided
                if train_query_ids is not None and qid not in train_query_ids:
                    continue

                # Only include positive judgments
                if rel > 0:
                    query_to_positive_pids[qid].append(pid)

    return dict(query_to_positive_pids)


def load_run_file(run_file, train_query_ids=None, top_k=1000):
    """
    Load run file to get candidate documents for negative sampling.

    Args:
        run_file: Path to run file
        train_query_ids: Optional list of query IDs to filter by
        top_k: Number of top documents to keep per query

    Returns:
        dict: Mapping from query IDs to lists of candidate passage IDs
    """
    query_to_candidates = defaultdict(list)

    print(f"Loading run file from {run_file}...")
    with open(run_file, 'r') as f:
        for line in tqdm(f, desc="Reading run file"):
            parts = line.strip().split()
            if len(parts) >= 6:  # TREC format: qid Q0 docid rank score run_name
                qid = parts[0]
                pid = parts[2]

                # Filter by train_query_ids if provided
                if train_query_ids is not None and qid not in train_query_ids:
                    continue

                # Check if we've already reached the limit for this query
                if len(query_to_candidates[qid]) < top_k:
                    query_to_candidates[qid].append(pid)

    return dict(query_to_candidates)


def create_training_triples(fold_idx, folds, query_embeddings, passage_embeddings,
                            qid_to_idx, pid_to_idx, qrels_file, run_file, num_negatives=1):
    """
    Create training triples for a specific fold.

    Args:
        fold_idx: Fold index to use
        folds: Loaded folds data
        query_embeddings: Query embeddings tensor
        passage_embeddings: Passage embeddings tensor
        qid_to_idx: Query ID to index mapping
        pid_to_idx: Passage ID to index mapping
        qrels_file: Path to qrels file
        run_file: Path to run file
        num_negatives: Number of negatives per positive

    Returns:
        list: List of training triples (query_idx, pos_idx, neg_idx)
    """
    # Get training query IDs for this fold
    train_query_ids = folds[str(fold_idx)]['training']

    print(f"Creating training triples for fold {fold_idx}")
    print(f"Training queries: {len(train_query_ids)}")

    # Load qrels to get positive passages
    query_to_positive_pids = load_qrels(qrels_file, train_query_ids)

    # Load run file for negative sampling
    query_to_candidates = load_run_file(run_file, train_query_ids)

    # Create training triples
    print("Creating training triples...")
    training_triples = []

    # Keep track of queries with at least one positive passage
    valid_qids = set()

    for qid in tqdm(train_query_ids, desc="Processing training queries"):
        # Skip if query has no relevant passages or is not in index
        if qid not in query_to_positive_pids or qid not in qid_to_idx:
            continue

        query_idx = qid_to_idx[qid]
        positive_pids = query_to_positive_pids[qid]

        # Filter out positive passages not in our index
        valid_positive_pids = [pid for pid in positive_pids if pid in pid_to_idx]

        if not valid_positive_pids:
            continue

        valid_qids.add(qid)

        # For each positive passage, create training triples with negatives
        for pos_pid in valid_positive_pids:
            pos_idx = pid_to_idx[pos_pid]

            # Generate negatives for this query-positive pair
            for _ in range(num_negatives):
                # Prefer to sample from candidate list if available
                if qid in query_to_candidates and query_to_candidates[qid]:
                    # Sample from candidates, filtering out positives
                    candidates = [pid for pid in query_to_candidates[qid]
                                  if pid in pid_to_idx and pid not in positive_pids]

                    if candidates:
                        neg_pid = random.choice(candidates)
                        neg_idx = pid_to_idx[neg_pid]
                    else:
                        # Fall back to random if no valid candidates
                        neg_idx = random.randint(0, len(pid_to_idx) - 1)
                else:
                    # Random negative sampling
                    neg_idx = random.randint(0, len(pid_to_idx) - 1)

                # Create a training triple
                training_triples.append((query_idx, pos_idx, neg_idx))

    print(f"Created {len(training_triples)} training triples from {len(valid_qids)} valid queries")
    return training_triples


def create_test_data(fold_idx, folds, qid_to_idx, pid_to_idx, qrels_file, run_file, max_candidates=1000):
    """
    Create test data for evaluation.

    Args:
        fold_idx: Fold index to use
        folds: Loaded folds data
        qid_to_idx: Query ID to index mapping
        pid_to_idx: Passage ID to index mapping
        qrels_file: Path to qrels file
        run_file: Path to run file
        max_candidates: Maximum number of candidates per query

    Returns:
        dict: Mapping from query IDs to lists of candidate passage IDs
    """
    # Get test query IDs for this fold
    test_query_ids = folds[str(fold_idx)]['testing']

    print(f"Creating test data for fold {fold_idx}")
    print(f"Test queries: {len(test_query_ids)}")

    # Load qrels to get positive passages
    query_to_positive_pids = load_qrels(qrels_file, test_query_ids)

    # Load run file for candidates
    query_to_candidates = load_run_file(run_file, test_query_ids)

    # Create final query to candidate mapping
    query_to_candidates_final = {}
    for qid in test_query_ids:
        if qid not in qid_to_idx:
            continue

        # Collect candidates
        candidates = set()

        # Add relevant documents first
        if qid in query_to_positive_pids:
            for pid in query_to_positive_pids[qid]:
                if pid in pid_to_idx:
                    candidates.add(pid)

        # Add candidates from run file
        if qid in query_to_candidates:
            for pid in query_to_candidates[qid]:
                if pid in pid_to_idx:
                    candidates.add(pid)
                    # Limit to max_candidates
                    if len(candidates) >= max_candidates:
                        break

        # Convert to list for final mapping
        if candidates:
            query_to_candidates_final[qid] = list(candidates)

    print(f"Created test data for {len(query_to_candidates_final)} valid test queries")
    return query_to_candidates_final


def create_all_triples(query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx, qrels_file, run_file,
                       num_negatives=1):
    """
    Create training triples for all queries without fold division.

    Args:
        query_embeddings: Query embeddings tensor
        passage_embeddings: Passage embeddings tensor
        qid_to_idx: Query ID to index mapping
        pid_to_idx: Passage ID to index mapping
        qrels_file: Path to qrels file
        run_file: Path to run file
        num_negatives: Number of negatives per positive

    Returns:
        list: List of training triples (query_idx, pos_idx, neg_idx)
    """
    # Load qrels to get positive passages
    query_to_positive_pids = load_qrels(qrels_file)

    # Load run file for negative sampling
    query_to_candidates = load_run_file(run_file)

    # Create training triples
    print("Creating training triples for all queries...")
    all_triples = []

    # Keep track of queries with at least one positive passage
    valid_qids = set()

    for qid in tqdm(query_to_positive_pids.keys(), desc="Processing all queries"):
        # Skip if query is not in index
        if qid not in qid_to_idx:
            continue

        query_idx = qid_to_idx[qid]
        positive_pids = query_to_positive_pids[qid]

        # Filter out positive passages not in our index
        valid_positive_pids = [pid for pid in positive_pids if pid in pid_to_idx]

        if not valid_positive_pids:
            continue

        valid_qids.add(qid)

        # For each positive passage, create training triples with negatives
        for pos_pid in valid_positive_pids:
            pos_idx = pid_to_idx[pos_pid]

            # Generate negatives for this query-positive pair
            for _ in range(num_negatives):
                # Prefer to sample from candidate list if available
                if qid in query_to_candidates and query_to_candidates[qid]:
                    # Sample from candidates, filtering out positives
                    candidates = [pid for pid in query_to_candidates[qid]
                                  if pid in pid_to_idx and pid not in positive_pids]

                    if candidates:
                        neg_pid = random.choice(candidates)
                        neg_idx = pid_to_idx[neg_pid]
                    else:
                        # Fall back to random if no valid candidates
                        neg_idx = random.randint(0, len(pid_to_idx) - 1)
                else:
                    # Random negative sampling
                    neg_idx = random.randint(0, len(pid_to_idx) - 1)

                # Create a training triple
                all_triples.append((query_idx, pos_idx, neg_idx))

    print(f"Created {len(all_triples)} training triples from {len(valid_qids)} valid queries")
    return all_triples


def save_training_triples(triples, output_path):
    """Save training triples to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving {len(triples)} training triples to {output_path}")
    torch.save(triples, output_path)
    print("Done!")


def save_test_data(test_data, output_path):
    """Save test data to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving test data for {len(test_data)} queries to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print("Done!")


def main():
    """Main function to create and save training triples for cross-validation."""
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create training triples for cross-validation training')
    parser.add_argument('--dataset', type=str, choices=['car', 'robust'], required=True,
                        help='Dataset to use (car or robust)')
    parser.add_argument('--qrels-file', type=str, default=None,
                        help='Path to qrels file (optional, defaults to config value)')
    parser.add_argument('--run-file', type=str, default=None,
                        help='Path to run file (optional, defaults to config value)')
    parser.add_argument('--folds-file', type=str, default=None,
                        help='Path to folds JSON file (optional, defaults to config value)')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='Specific folds to process (default: all folds)')
    parser.add_argument('--negatives', type=int, default=1,
                        help='Number of negative samples per positive (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for triplets files (defaults to config value)')
    parser.add_argument('--all-triples', action='store_true',
                        help='Create a single file with all triples regardless of folds')
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    config_prefix = dataset_name.upper()

    # Set defaults from config if not provided
    qrels_file = args.qrels_file or getattr(config, f"{config_prefix}_QRELS_FILE")
    run_file = args.run_file or getattr(config, f"{config_prefix}_RUN_FILE")
    folds_file = args.folds_file or getattr(config, f"{config_prefix}_FOLDS_FILE")
    output_dir = args.output_dir or os.path.join(os.path.dirname(config.MODEL_SAVE_DIR), "cv_triples", dataset_name)
    num_negatives = args.negatives

    print(f"Creating training triples for dataset: {dataset_name}")
    print(f"Qrels file: {qrels_file}")
    print(f"Run file: {run_file}")
    print(f"Folds file: {folds_file}")
    print(f"Number of negatives per positive: {num_negatives}")
    print(f"Output directory: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load embeddings and mappings
    print(f"Loading embeddings for {dataset_name}...")
    query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx = load_embeddings_and_mappings(dataset_name)

    print(f"Loaded {query_embeddings.shape[0]} query embeddings")
    print(f"Loaded {passage_embeddings.shape[0]} passage embeddings")

    # Create and save ALL triples if requested (non-fold specific)
    if args.all_triples:
        print("\n==== Creating ALL training triples (non-fold specific) ====")
        all_triples = create_all_triples(
            query_embeddings,
            passage_embeddings,
            qid_to_idx,
            pid_to_idx,
            qrels_file,
            run_file,
            num_negatives=num_negatives
        )
        all_triples_path = os.path.join(output_dir, f"{dataset_name}_all_triples.pt")
        save_training_triples(all_triples, all_triples_path)

    # Load folds
    print(f"Loading folds from {folds_file}")
    folds = load_folds(folds_file)
    print(f"Loaded {len(folds)} folds")

    # Determine which folds to process
    fold_indices = args.folds if args.folds else sorted([int(k) for k in folds.keys()])
    print(f"Processing folds: {fold_indices}")

    # Process each fold
    for fold_idx in fold_indices:
        print(f"\n{'=' * 50}")
        print(f"Processing fold {fold_idx}")
        print(f"{'=' * 50}")

        # Create training triples
        training_triples = create_training_triples(
            fold_idx,
            folds,
            query_embeddings,
            passage_embeddings,
            qid_to_idx,
            pid_to_idx,
            qrels_file,
            run_file,
            num_negatives=num_negatives
        )

        # Save training triples
        triples_output_path = os.path.join(output_dir, f"fold_{fold_idx}_triples.pt")
        save_training_triples(training_triples, triples_output_path)

        # Create and save test data
        test_data = create_test_data(
            fold_idx,
            folds,
            qid_to_idx,
            pid_to_idx,
            qrels_file,
            run_file
        )

        test_data_output_path = os.path.join(output_dir, f"fold_{fold_idx}_test_data.json")
        save_test_data(test_data, test_data_output_path)

    print("\nAll processing completed successfully!")


if __name__ == "__main__":
    main()