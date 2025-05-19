#!/usr/bin/env python
# preprocess_embeddings.py
import os
import numpy as np
import json
import ir_datasets
import requests
import tarfile
import io
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import time

# Import config for output paths
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure config.py is in the same directory.")
    exit(1)


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


def collect_unique_ids():
    """
    Collect all unique query IDs and passage IDs that we need embeddings for
    using ir_datasets.
    """
    print("Collecting unique query and passage IDs...")

    unique_qids = set()
    unique_pids = set()

    # From training triples
    print("Reading training triples from ir_datasets...")
    train_dataset = ir_datasets.load("msmarco-passage/train/triples-small")

    # Add query IDs from training dataset
    print("Getting query IDs from training dataset...")
    for query in tqdm(train_dataset.queries_iter(), desc="Train queries"):
        unique_qids.add(query.query_id)

    # Add passage pairs from training dataset
    print("Getting passage IDs from training triples...")
    for docpair in tqdm(train_dataset.docpairs_iter(), desc="Train docpairs"):
        unique_qids.add(docpair.query_id)
        unique_pids.add(docpair.doc_id_a)
        unique_pids.add(docpair.doc_id_b)

    # From dev queries and qrels
    print("Loading dev dataset...")
    dev_dataset = ir_datasets.load("msmarco-passage/dev/small")

    print("Getting query IDs from dev dataset...")
    for query in tqdm(dev_dataset.queries_iter(), desc="Dev queries"):
        unique_qids.add(query.query_id)

    print("Getting passage IDs from dev qrels...")
    for qrel in tqdm(dev_dataset.qrels_iter(), desc="Dev qrels"):
        unique_pids.add(qrel.doc_id)

    # We need to handle top1000 candidates separately since ir_datasets doesn't provide this directly
    print("Reading top1000 dev candidates...")
    top1000_path = download_top1000_dev("temp_download")

    if top1000_path and os.path.exists(top1000_path):
        with open(top1000_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Top1000 candidates"):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Format: qid Q0 pid rank score run_name
                        pid = parts[2]
                        unique_pids.add(pid)
                except:
                    continue

    print(f"Found {len(unique_qids)} unique query IDs")
    print(f"Found {len(unique_pids)} unique passage IDs")

    return unique_qids, unique_pids


def load_texts(unique_qids, unique_pids):
    """
    Load the actual text for the collected IDs using ir_datasets.
    """
    print("Loading query and passage texts...")

    qid_to_text = {}
    pid_to_text = {}

    # Load base dataset for passages
    print("Loading base dataset...")
    base_dataset = ir_datasets.load("msmarco-passage")

    # Load train and dev datasets for queries
    train_dataset = ir_datasets.load("msmarco-passage/train/triples-small")
    dev_dataset = ir_datasets.load("msmarco-passage/dev/small")

    # Load query texts from train dataset
    print("Loading query texts from train dataset...")
    for query in tqdm(train_dataset.queries_iter(), desc="Train queries"):
        if query.query_id in unique_qids:
            qid_to_text[query.query_id] = query.text

    # Load query texts from dev dataset
    print("Loading query texts from dev dataset...")
    for query in tqdm(dev_dataset.queries_iter(), desc="Dev queries"):
        if query.query_id in unique_qids:
            qid_to_text[query.query_id] = query.text

    # Load passage texts from base dataset
    print("Loading passage texts from dataset...")
    # Create a list of doc_ids we need to fetch to avoid unnecessary iteration
    pids_to_fetch = list(unique_pids)

    # Print warning about potential long processing time
    print(f"Fetching {len(pids_to_fetch)} passages from dataset. This may take a while...")

    # Process documents in batches to show progress
    total_docs = base_dataset.docs_count()
    batch_size = 100000  # Process in batches to show progress

    print(f"Scanning through {total_docs} documents to find matching IDs...")

    # Create a set for faster lookups
    unique_pids_set = set(unique_pids)
    remaining_pids = set(unique_pids)

    # Process documents and update progress every batch
    doc_iter = base_dataset.docs_iter()
    processed = 0

    with tqdm(total=total_docs, desc="Processing documents") as pbar:
        try:
            while True:
                for _ in range(batch_size):
                    doc = next(doc_iter)
                    processed += 1

                    # Check if this doc is in our target set
                    if doc.doc_id in unique_pids_set:
                        pid_to_text[doc.doc_id] = doc.text
                        remaining_pids.discard(doc.doc_id)

                        # If we've found all PIDs, we can exit early
                        if not remaining_pids:
                            raise StopIteration

                # Update progress bar
                pbar.update(batch_size)
                pbar.set_postfix({"found": len(pid_to_text), "remaining": len(remaining_pids)})

        except StopIteration:
            # Update final progress
            pbar.update(pbar.total - pbar.n)
            pbar.set_postfix({"found": len(pid_to_text), "remaining": len(remaining_pids)})

    print(f"Loaded text for {len(qid_to_text)} queries")
    print(f"Loaded text for {len(pid_to_text)} passages")

    # Filter IDs to only those we have text for
    unique_qids = set(qid_to_text.keys())
    unique_pids = set(pid_to_text.keys())

    # Log warnings about missing texts
    missing_qids = set(unique_qids) - set(qid_to_text.keys())
    missing_pids = set(unique_pids) - set(pid_to_text.keys())

    if missing_qids:
        print(f"Warning: Couldn't find text for {len(missing_qids)} query IDs")

    if missing_pids:
        print(f"Warning: Couldn't find text for {len(missing_pids)} passage IDs")

    return unique_qids, unique_pids, qid_to_text, pid_to_text


def generate_embeddings(qid_to_text, pid_to_text, model_name=None, device=None):
    """
    Generate embeddings for all queries and passages.
    """
    if model_name is None:
        model_name = config.SBERT_MODEL_NAME

    if device is None:
        device = config.DEVICE

    print(f"Loading SBERT model: {model_name}...")
    sbert_model = SentenceTransformer(model_name, device=device)

    # Sort IDs for consistent indexing
    sorted_qids = sorted(list(qid_to_text.keys()))
    sorted_pids = sorted(list(pid_to_text.keys()))

    # Create mappings
    query_id_to_idx = {qid: i for i, qid in enumerate(sorted_qids)}
    passage_id_to_idx = {pid: i for i, pid in enumerate(sorted_pids)}

    # Extract text lists
    query_texts_list = [qid_to_text[qid] for qid in sorted_qids]
    passage_texts_list = [pid_to_text[pid] for pid in sorted_pids]

    # Generate query embeddings
    print(f"Encoding {len(query_texts_list)} query texts...")
    query_embeddings_np = sbert_model.encode(
        query_texts_list,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Keep raw embeddings
    )

    # Generate passage embeddings
    print(f"Encoding {len(passage_texts_list)} passage texts...")
    passage_embeddings_np = sbert_model.encode(
        passage_texts_list,
        batch_size=128,  # Smaller batch size for longer passages
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Keep raw embeddings
    )

    print(f"Query embeddings shape: {query_embeddings_np.shape}")
    print(f"Passage embeddings shape: {passage_embeddings_np.shape}")

    return query_embeddings_np, passage_embeddings_np, query_id_to_idx, passage_id_to_idx


def save_embeddings_and_mappings(query_embeddings_np, passage_embeddings_np,
                                 query_id_to_idx, passage_id_to_idx, model_name):
    """
    Save embeddings and mapping files using paths from config.
    """
    print("Saving embeddings and mappings...")

    # Create directory if it doesn't exist
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)

    # Save embeddings
    print(f"Saving query embeddings to {config.QUERY_EMBEDDINGS_PATH}...")
    np.save(config.QUERY_EMBEDDINGS_PATH, query_embeddings_np)

    print(f"Saving passage embeddings to {config.PASSAGE_EMBEDDINGS_PATH}...")
    np.save(config.PASSAGE_EMBEDDINGS_PATH, passage_embeddings_np)

    # Save ID mappings
    print(f"Saving query ID to index mapping to {config.QUERY_ID_TO_IDX_PATH}...")
    with open(config.QUERY_ID_TO_IDX_PATH, 'w') as f:
        json.dump(query_id_to_idx, f, indent=2)

    print(f"Saving passage ID to index mapping to {config.PASSAGE_ID_TO_IDX_PATH}...")
    with open(config.PASSAGE_ID_TO_IDX_PATH, 'w') as f:
        json.dump(passage_id_to_idx, f, indent=2)

    # Save metadata
    metadata = {
        'num_queries': len(query_id_to_idx),
        'num_passages': len(passage_id_to_idx),
        'embedding_dim': query_embeddings_np.shape[1],
        'model_name': model_name,
        'generated_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(config.EMBEDDING_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("All files saved successfully!")


def verify_embeddings():
    """
    Quick verification of saved embeddings.
    """
    print("Verifying saved embeddings...")

    # Check if files exist
    files_to_check = [
        config.QUERY_EMBEDDINGS_PATH,
        config.PASSAGE_EMBEDDINGS_PATH,
        config.QUERY_ID_TO_IDX_PATH,
        config.PASSAGE_ID_TO_IDX_PATH
    ]

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found")
            return False

    # Load and check embeddings
    print("Loading saved query embeddings...")
    query_embeddings = np.load(config.QUERY_EMBEDDINGS_PATH)

    print("Loading saved passage embeddings...")
    passage_embeddings = np.load(config.PASSAGE_EMBEDDINGS_PATH)

    print("Loading query ID to index mapping...")
    with open(config.QUERY_ID_TO_IDX_PATH, 'r') as f:
        query_id_to_idx = json.load(f)

    print("Loading passage ID to index mapping...")
    with open(config.PASSAGE_ID_TO_IDX_PATH, 'r') as f:
        passage_id_to_idx = json.load(f)

    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Passage embeddings shape: {passage_embeddings.shape}")
    print(f"Number of query IDs: {len(query_id_to_idx)}")
    print(f"Number of passage IDs: {len(passage_id_to_idx)}")

    # Verify consistency
    assert query_embeddings.shape[0] == len(query_id_to_idx), "Query embedding count mismatch"
    assert passage_embeddings.shape[0] == len(passage_id_to_idx), "Passage embedding count mismatch"

    print("Verification successful!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for MS MARCO passages and queries using ir_datasets')
    parser.add_argument('--model-name', type=str, default=None,
                        help=f'SBERT model name (default: {config.SBERT_MODEL_NAME})')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                        help=f'Device to use for encoding (cuda or cpu, default: {config.DEVICE})')
    parser.add_argument('--skip-if-exists', action='store_true',
                        help='Skip generation if embeddings already exist')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing embeddings')
    parser.add_argument('--limit-passages', type=int, default=None,
                        help='Limit number of passages to process (for testing)')
    parser.add_argument('--limit-queries', type=int, default=None,
                        help='Limit number of queries to process (for testing)')

    args = parser.parse_args()

    # Use default model name from config if not specified
    model_name = args.model_name if args.model_name else config.SBERT_MODEL_NAME

    # Check if embeddings already exist
    if args.skip_if_exists and all(os.path.exists(f) for f in [
        config.QUERY_EMBEDDINGS_PATH,
        config.PASSAGE_EMBEDDINGS_PATH,
        config.QUERY_ID_TO_IDX_PATH,
        config.PASSAGE_ID_TO_IDX_PATH
    ]):
        print("Embeddings already exist. Skipping generation.")
        if verify_embeddings():
            print("Existing embeddings are valid.")
        return

    # Verify only mode
    if args.verify_only:
        verify_embeddings()
        return

    # Main preprocessing pipeline
    print("Starting embedding generation...")
    start_time = time.time()

    # Step 1: Collect all IDs we need
    unique_qids, unique_pids = collect_unique_ids()

    # Apply limits if specified (for testing)
    if args.limit_queries and len(unique_qids) > args.limit_queries:
        print(f"Limiting to {args.limit_queries} queries for testing")
        unique_qids = set(list(unique_qids)[:args.limit_queries])

    if args.limit_passages and len(unique_pids) > args.limit_passages:
        print(f"Limiting to {args.limit_passages} passages for testing")
        unique_pids = set(list(unique_pids)[:args.limit_passages])

    # Step 2: Load texts for these IDs
    unique_qids, unique_pids, qid_to_text, pid_to_text = load_texts(unique_qids, unique_pids)

    # Step 3: Generate embeddings
    query_embeddings, passage_embeddings, query_id_to_idx, passage_id_to_idx = generate_embeddings(
        qid_to_text, pid_to_text, model_name, args.device
    )

    # Step 4: Save everything
    save_embeddings_and_mappings(query_embeddings, passage_embeddings,
                                 query_id_to_idx, passage_id_to_idx, model_name)

    # Step 5: Verify
    verify_embeddings()

    # Clean up temporary files
    temp_dir = os.path.join(os.path.dirname(config.EMBEDDING_DIR), 'temp_download')
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("Embedding generation complete!")


if __name__ == "__main__":
    main()