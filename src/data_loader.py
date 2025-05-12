# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
from tqdm import tqdm
import config
import os


class MSMARCOTriplesDataset(Dataset):
    def __init__(self, triples_path, qid_to_idx, pid_to_idx, query_embeddings, passage_embeddings, limit_size=None):
        self.qid_to_idx = qid_to_idx
        self.pid_to_idx = pid_to_idx
        self.query_embeddings = query_embeddings
        self.passage_embeddings = passage_embeddings
        self.triples = []

        print(f"Loading triples from {triples_path}...")
        with open(triples_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                if limit_size and i >= limit_size:
                    print(f"Reached limit of {limit_size} triples.")
                    break
                try:
                    qid, pos_pid, neg_pid = line.strip().split('\t')
                    # Ensure IDs are in our mapping (and thus have embeddings)
                    if qid in self.qid_to_idx and \
                            pos_pid in self.pid_to_idx and \
                            neg_pid in self.pid_to_idx:
                        self.triples.append((qid, pos_pid, neg_pid))
                except ValueError:
                    # Handle cases where line might not have 3 parts, or other parsing issues
                    continue  # Skip this line
        print(f"Loaded {len(self.triples)} valid triples.")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, pos_pid, neg_pid = self.triples[idx]

        q_embed_idx = self.qid_to_idx[qid]
        pos_p_embed_idx = self.pid_to_idx[pos_pid]
        neg_p_embed_idx = self.pid_to_idx[neg_pid]

        q_embed = torch.tensor(self.query_embeddings[q_embed_idx], dtype=torch.float)
        pos_p_embed = torch.tensor(self.passage_embeddings[pos_p_embed_idx], dtype=torch.float)
        neg_p_embed = torch.tensor(self.passage_embeddings[neg_p_embed_idx], dtype=torch.float)

        return q_embed, pos_p_embed, neg_p_embed


def load_embeddings_and_mappings():
    print("Loading embeddings and ID mappings...")

    # Check if files exist
    if not os.path.exists(config.QUERY_EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Query embeddings not found at {config.QUERY_EMBEDDINGS_PATH}. Please run preprocess_embeddings.py first.")

    query_embeddings = np.load(config.QUERY_EMBEDDINGS_PATH)
    passage_embeddings = np.load(config.PASSAGE_EMBEDDINGS_PATH)

    with open(config.QUERY_ID_TO_IDX_PATH, 'r') as f:
        qid_to_idx = json.load(f)
    with open(config.PASSAGE_ID_TO_IDX_PATH, 'r') as f:
        pid_to_idx = json.load(f)

    print(f"Loaded {query_embeddings.shape[0]} query embeddings.")
    print(f"Loaded {passage_embeddings.shape[0]} passage embeddings.")
    return query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx


def load_dev_data_for_eval(dev_queries_path, dev_candidates_path, qid_to_idx, pid_to_idx):
    """
    Loads dev queries and their top-K candidates for evaluation.
    Returns a dictionary: {qid: [list of candidate pids]}
    """
    dev_queries = {}

    # Load dev queries that have embeddings
    if os.path.exists(dev_queries_path):
        with open(dev_queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    if qid in qid_to_idx:  # Only keep queries for which we have embeddings
                        dev_queries[qid] = []

    # Load candidates (e.g., from top1000.dev file format: qid Q0 pid rank score run_name)
    query_to_candidates = {}

    if os.path.exists(dev_candidates_path):
        with open(dev_candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    qid = parts[0]
                    pid = parts[2]

                    if qid not in query_to_candidates:
                        query_to_candidates[qid] = []

                    if pid in pid_to_idx:  # Only keep passages for which we have embeddings
                        query_to_candidates[qid].append(pid)

    # Filter queries that might not have candidates or embeddings
    valid_dev_queries = {
        qid: candidates
        for qid, candidates in query_to_candidates.items()
        if qid in dev_queries and candidates  # Ensure query has candidates
    }
    print(f"Loaded candidates for {len(valid_dev_queries)} dev queries.")
    return valid_dev_queries


def create_msmarco_train_dataloader(limit_size=None):
    """Helper function to create train dataloader with loaded embeddings"""
    query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx = load_embeddings_and_mappings()

    train_dataset = MSMARCOTriplesDataset(
        config.TRAIN_TRIPLES_PATH,
        qid_to_idx,
        pid_to_idx,
        query_embeddings,
        passage_embeddings,
        limit_size=limit_size
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.DEVICE == "cuda" else False
    )

    return train_dataloader, query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx