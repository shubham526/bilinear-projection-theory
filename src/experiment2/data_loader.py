#!/usr/bin/env python
# data_loader.py
"""
Data loading utilities using ir_datasets for MS MARCO passage ranking
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import ir_datasets
from tqdm import tqdm
from collections import defaultdict
import requests
import tarfile
import io
import config


# In data_loader.py, update the load_embeddings_and_mappings function

def load_embeddings_and_mappings(dataset_name=None, query_emb_path=None, passage_emb_path=None,
                                 qid_map_path=None, pid_map_path=None):
    """
    Load embeddings and ID mappings from the paths specified in config.

    Args:
        dataset_name: Optional dataset name (car, robust, or None for default paths)
        query_emb_path, passage_emb_path, qid_map_path, pid_map_path: Optional explicit paths
    """
    print(f"Loading embeddings and ID mappings for {dataset_name or 'default'}...")

    # Determine paths based on dataset
    if not query_emb_path:
        if dataset_name:
            config_prefix = dataset_name.upper()
            query_emb_path = getattr(config, f"{config_prefix}_QUERY_EMBEDDINGS_PATH",
                                     config.QUERY_EMBEDDINGS_PATH)
        else:
            query_emb_path = config.QUERY_EMBEDDINGS_PATH

    if not passage_emb_path:
        if dataset_name:
            config_prefix = dataset_name.upper()
            passage_emb_path = getattr(config, f"{config_prefix}_PASSAGE_EMBEDDINGS_PATH",
                                       config.PASSAGE_EMBEDDINGS_PATH)
        else:
            passage_emb_path = config.PASSAGE_EMBEDDINGS_PATH

    if not qid_map_path:
        if dataset_name:
            config_prefix = dataset_name.upper()
            qid_map_path = getattr(config, f"{config_prefix}_QUERY_ID_TO_IDX_PATH",
                                   config.QUERY_ID_TO_IDX_PATH)
        else:
            qid_map_path = config.QUERY_ID_TO_IDX_PATH

    if not pid_map_path:
        if dataset_name:
            config_prefix = dataset_name.upper()
            pid_map_path = getattr(config, f"{config_prefix}_PASSAGE_ID_TO_IDX_PATH",
                                   config.PASSAGE_ID_TO_IDX_PATH)
        else:
            pid_map_path = config.PASSAGE_ID_TO_IDX_PATH

    # Check if files exist
    for file_path in [query_emb_path, passage_emb_path, qid_map_path, pid_map_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load embeddings and ensure they're float32
    query_embeddings = np.load(query_emb_path).astype(np.float32)
    passage_embeddings = np.load(passage_emb_path).astype(np.float32)

    with open(qid_map_path, 'r') as f:
        qid_to_idx = json.load(f)
    with open(pid_map_path, 'r') as f:
        pid_to_idx = json.load(f)

    print(f"Loaded {query_embeddings.shape[0]} query embeddings (dtype: {query_embeddings.dtype})")
    print(f"Loaded {passage_embeddings.shape[0]} passage embeddings (dtype: {passage_embeddings.dtype})")

    return query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx


class MSMARCOTriplesDataset(Dataset):
    """
    Dataset for MS MARCO triples using precomputed embeddings.
    Can load directly from a file path (original style) or from ir_datasets.
    """

    def __init__(self, qid_to_idx, pid_to_idx, query_embeddings, passage_embeddings,
                 triples_path=None, use_ir_datasets=True, limit_size=None):
        """
        Initialize the dataset.

        Args:
            qid_to_idx: Mapping from query IDs to embedding indices
            pid_to_idx: Mapping from passage IDs to embedding indices
            query_embeddings: Query embeddings matrix (numpy array)
            passage_embeddings: Passage embeddings matrix (numpy array)
            triples_path: Path to triples file (only needed if use_ir_datasets=False)
            use_ir_datasets: Whether to load triples from ir_datasets (True) or file (False)
            limit_size: Optional size limit for testing (number of triples)
        """
        self.qid_to_idx = qid_to_idx
        self.pid_to_idx = pid_to_idx
        self.query_embeddings = query_embeddings
        self.passage_embeddings = passage_embeddings
        self.triples = []

        if use_ir_datasets:
            self._load_triples_from_ir_datasets(limit_size)
        else:
            if not triples_path:
                raise ValueError("triples_path must be provided when use_ir_datasets=False")
            self._load_triples_from_file(triples_path, limit_size)

        print(f"Loaded {len(self.triples)} valid triples.")

    def _load_triples_from_file(self, triples_path, limit_size):
        """Load triples from a file"""
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
                        self.triples.append((
                            self.qid_to_idx[qid],
                            self.pid_to_idx[pos_pid],
                            self.pid_to_idx[neg_pid]
                        ))
                except ValueError:
                    # Handle cases where line might not have 3 parts, or other parsing issues
                    continue  # Skip this line

    def _load_triples_from_ir_datasets(self, limit_size):
        """Load triples from ir_datasets"""
        print("Loading training triples from ir_datasets...")

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
                    print(f"Reached limit of {limit_size} triples.")
                    break
            else:
                skipped += 1

        print(f"Skipped {skipped} triples due to missing embeddings")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qidx, pos_pidx, neg_pidx = self.triples[idx]

        q_embed = torch.tensor(self.query_embeddings[qidx], dtype=torch.float)
        pos_p_embed = torch.tensor(self.passage_embeddings[pos_pidx], dtype=torch.float)
        neg_p_embed = torch.tensor(self.passage_embeddings[neg_pidx], dtype=torch.float)

        return q_embed, pos_p_embed, neg_p_embed


def download_top1000_dev(temp_dir=None):
    """
    Download and extract top1000.dev file if needed.
    """
    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(config.EMBEDDING_DIR), 'temp_download')

    os.makedirs(temp_dir, exist_ok=True)
    top1000_path = os.path.join(temp_dir, 'top1000.dev.tsv')

    # Check if file already exists
    if os.path.exists(top1000_path):
        print(f"Top1000 dev file already exists: {top1000_path}")
        return top1000_path

    print("Downloading top1000.dev.tar.gz...")
    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract directly from the stream
        tar_bytes = io.BytesIO(response.content)
        with tarfile.open(fileobj=tar_bytes, mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.name == 'top1000.dev':
                    f = tar.extractfile(member)
                    if f:
                        with open(top1000_path, 'wb') as out_file:
                            out_file.write(f.read())
                        break

        print(f"Downloaded and extracted top1000.dev to: {top1000_path}")
        return top1000_path

    except Exception as e:
        print(f"Error downloading top1000.dev: {e}")
        return None


def load_dev_data_for_eval(qid_to_idx, pid_to_idx, use_ir_datasets=True, dataset_name=None):
    """
    Loads dev queries and their top-K candidates for evaluation.
    Can use either ir_datasets (preferred) or config-defined paths.

    Args:
        qid_to_idx: Query ID to index mapping
        pid_to_idx: Passage ID to index mapping
        use_ir_datasets: Whether to use ir_datasets for loading
        dataset_name: Optional dataset name (msmarco-passage by default)

    Returns:
        A dictionary: {qid: [(pid, qidx, pidx), ...]}
    """
    if use_ir_datasets:
        return _load_dev_data_from_ir_datasets(qid_to_idx, pid_to_idx)
    else:
        return _load_dev_data_from_files(config.DEV_QUERIES_PATH, config.DEV_CANDIDATES_PATH, qid_to_idx, pid_to_idx)


def _load_dev_data_from_files(dev_queries_path, dev_candidates_path, qid_to_idx, pid_to_idx):
    """
    Loads dev queries and their top-K candidates from files.
    Returns a dictionary: {qid: [(pid, qidx, pidx), ...]}
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
                        query_to_candidates[qid].append((pid, qid_to_idx[qid], pid_to_idx[pid]))

    # Filter queries that might not have candidates or embeddings
    valid_dev_queries = {
        qid: candidates
        for qid, candidates in query_to_candidates.items()
        if qid in dev_queries and candidates  # Ensure query has candidates
    }
    print(f"Loaded candidates for {len(valid_dev_queries)} dev queries.")
    return valid_dev_queries


def _load_dev_data_from_ir_datasets(qid_to_idx, pid_to_idx):
    """
    Load dev data for evaluation using ir_datasets.
    Returns a dictionary: {qid: [(pid, qidx, pidx), ...]}
    """
    print("Loading dev data using ir_datasets...")

    # Load dev dataset
    dev_dataset = ir_datasets.load("msmarco-passage/dev/small")

    # Create dictionary mapping each query to its candidates
    dev_query_to_candidates = defaultdict(list)

    # Get valid query IDs (those with embeddings)
    valid_qids = set()
    for query in dev_dataset.queries_iter():
        if query.query_id in qid_to_idx:
            valid_qids.add(query.query_id)

    print(f"Found {len(valid_qids)} valid queries in dev set")

    # Process scored documents directly from ir_datasets
    processed_count = 0
    skipped_count = 0

    print("Loading scored documents from ir_datasets...")
    for scoreddoc in tqdm(dev_dataset.scoreddocs_iter(), desc="Loading candidate passages"):
        qid = scoreddoc.query_id
        pid = scoreddoc.doc_id

        # Only process if this is a valid query
        if qid in valid_qids:
            # Check if the passage ID is in our embeddings
            if pid in pid_to_idx:
                # Store candidate as tuple (passage_id, query_embedding_idx, passage_embedding_idx)
                dev_query_to_candidates[qid].append(
                    (pid, qid_to_idx[qid], pid_to_idx[pid])
                )
                processed_count += 1
            else:
                skipped_count += 1

    print(f"Processed {processed_count} candidate passages")
    print(f"Skipped {skipped_count} passages due to missing embeddings")
    print(f"Loaded candidates for {len(dev_query_to_candidates)} queries")

    # Print some stats about the number of candidates per query
    if dev_query_to_candidates:
        candidate_counts = [len(candidates) for candidates in dev_query_to_candidates.values()]
        avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
        print(f"Average candidates per query: {avg_candidates:.1f}")
        print(f"Min candidates: {min(candidate_counts) if candidate_counts else 0}")
        print(f"Max candidates: {max(candidate_counts) if candidate_counts else 0}")

        # Print a few examples
        sample_count = min(3, len(dev_query_to_candidates))
        print(f"\nExample queries and candidate counts:")
        for i, (qid, candidates) in enumerate(list(dev_query_to_candidates.items())[:sample_count]):
            print(f"  Query {qid}: {len(candidates)} candidates")

    return dict(dev_query_to_candidates)


def create_msmarco_train_dataloader(query_embeddings=None, passage_embeddings=None,
                                    qid_to_idx=None, pid_to_idx=None,
                                    limit_size=None, use_ir_datasets=True, dataset_name=None):
    """
    Helper function to create train dataloader with loaded embeddings.
    Can use either ir_datasets or file-based loading.

    Args:
        query_embeddings: Optional pre-loaded query embeddings
        passage_embeddings: Optional pre-loaded passage embeddings
        qid_to_idx: Optional pre-loaded query ID to index mapping
        pid_to_idx: Optional pre-loaded passage ID to index mapping
        limit_size: Optional size limit for testing
        use_ir_datasets: Whether to use ir_datasets for loading
        dataset_name: Optional dataset name for loading embeddings

    Returns:
        train_dataloader: DataLoader for training
        train_dataset: Dataset object
        query_embeddings: Query embeddings
        passage_embeddings: Passage embeddings
        mappings: Tuple of (qid_to_idx, pid_to_idx)
    """
    # Load embeddings and mappings if not provided
    if query_embeddings is None or passage_embeddings is None or qid_to_idx is None or pid_to_idx is None:
        query_embeddings, passage_embeddings, qid_to_idx, pid_to_idx = load_embeddings_and_mappings(dataset_name)

    if use_ir_datasets:
        train_dataset = MSMARCOTriplesDataset(
            qid_to_idx,
            pid_to_idx,
            query_embeddings,
            passage_embeddings,
            use_ir_datasets=True,
            limit_size=limit_size
        )
    else:
        train_dataset = MSMARCOTriplesDataset(
            qid_to_idx,
            pid_to_idx,
            query_embeddings,
            passage_embeddings,
            triples_path=config.TRAIN_TRIPLES_PATH,
            use_ir_datasets=False,
            limit_size=limit_size
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(config, 'NUM_WORKERS', 4),
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, train_dataset, query_embeddings, passage_embeddings, (qid_to_idx, pid_to_idx)