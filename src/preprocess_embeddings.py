# preprocess_embeddings.py
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import collections
import argparse
import time

# Import config
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure config.py is in the same directory.")
    exit(1)


def collect_ids_from_files():
    """
    Collect all unique query IDs and passage IDs that we need embeddings for.
    """
    print("Collecting unique query and passage IDs...")

    unique_qids = set()
    unique_pids = set()

    # From training triples
    print("Reading training triples...")
    if os.path.exists(config.TRAIN_TRIPLES_PATH):
        with open(config.TRAIN_TRIPLES_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Training triples"):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        qid, pos_pid, neg_pid = parts[:3]
                        unique_qids.add(qid)
                        unique_pids.add(pos_pid)
                        unique_pids.add(neg_pid)
                except:
                    continue
    else:
        print(f"Warning: {config.TRAIN_TRIPLES_PATH} not found")

    # From dev queries
    print("Reading dev queries...")
    if os.path.exists(config.DEV_QUERIES_PATH):
        with open(config.DEV_QUERIES_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Dev queries"):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        qid = parts[0]
                        unique_qids.add(qid)
                except:
                    continue
    else:
        print(f"Warning: {config.DEV_QUERIES_PATH} not found")

    # From dev candidates
    print("Reading dev candidates...")
    if os.path.exists(config.DEV_CANDIDATES_PATH):
        with open(config.DEV_CANDIDATES_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Dev candidates"):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Format: qid Q0 pid rank score run_name
                        pid = parts[2]
                        unique_pids.add(pid)
                except:
                    continue
    else:
        print(f"Warning: {config.DEV_CANDIDATES_PATH} not found")

    print(f"Found {len(unique_qids)} unique query IDs")
    print(f"Found {len(unique_pids)} unique passage IDs")

    return unique_qids, unique_pids


def load_texts(unique_qids, unique_pids):
    """
    Load the actual text for the collected IDs.
    """
    print("Loading query and passage texts...")

    qid_to_text = {}
    pid_to_text = {}

    # Load query texts
    print("Loading query texts...")
    query_files_to_check = [
        os.path.join(config.MSMARCO_V1_DIR, "queries.train.tsv"),
        os.path.join(config.MSMARCO_V1_DIR, "queries.dev.small.tsv"),
        os.path.join(config.MSMARCO_V1_DIR, "queries.dev.tsv"),
        config.DEV_QUERIES_PATH,
    ]

    # Remove duplicates and filter existing files
    query_files_to_check = list(dict.fromkeys(query_files_to_check))
    query_files_to_check = [f for f in query_files_to_check if os.path.exists(f)]

    for q_file in query_files_to_check:
        print(f"Reading {q_file}...")
        with open(q_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(q_file)}"):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], parts[1]
                        if qid in unique_qids:
                            qid_to_text[qid] = qtext
                except:
                    continue

    # Load passage texts from collection
    print("Loading passage texts from collection...")
    if os.path.exists(config.CORPUS_PATH):
        with open(config.CORPUS_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading collection.tsv"):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        pid = parts[0]
                        ptext = parts[1]
                        if pid in unique_pids:
                            pid_to_text[pid] = ptext
                except:
                    continue
    else:
        print(f"Warning: {config.CORPUS_PATH} not found")

    print(f"Loaded text for {len(qid_to_text)} queries")
    print(f"Loaded text for {len(pid_to_text)} passages")

    # Filter IDs to only those we have text for
    unique_qids = set(qid_to_text.keys())
    unique_pids = set(pid_to_text.keys())

    return unique_qids, unique_pids, qid_to_text, pid_to_text


def generate_embeddings(qid_to_text, pid_to_text, model_name=None):
    """
    Generate embeddings for all queries and passages.
    """
    if model_name is None:
        model_name = config.SBERT_MODEL_NAME

    print(f"Loading SBERT model: {model_name}...")
    sbert_model = SentenceTransformer(model_name, device=config.DEVICE)

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
                                 query_id_to_idx, passage_id_to_idx):
    """
    Save embeddings and mapping files.
    """
    print("Saving embeddings and mappings...")

    # Create directory if it doesn't exist
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)

    # Save embeddings
    print("Saving query embeddings...")
    np.save(config.QUERY_EMBEDDINGS_PATH, query_embeddings_np)

    print("Saving passage embeddings...")
    np.save(config.PASSAGE_EMBEDDINGS_PATH, passage_embeddings_np)

    # Save ID mappings
    print("Saving query ID to index mapping...")
    with open(config.QUERY_ID_TO_IDX_PATH, 'w') as f:
        json.dump(query_id_to_idx, f, indent=2)

    print("Saving passage ID to index mapping...")
    with open(config.PASSAGE_ID_TO_IDX_PATH, 'w') as f:
        json.dump(passage_id_to_idx, f, indent=2)

    # Save metadata
    metadata = {
        'num_queries': len(query_id_to_idx),
        'num_passages': len(passage_id_to_idx),
        'embedding_dim': query_embeddings_np.shape[1],
        'model_name': config.SBERT_MODEL_NAME,
        'generated_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(config.EMBEDDING_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("All files saved successfully!")
    print(f"Embeddings saved to: {config.EMBEDDING_DIR}")


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
    query_embeddings = np.load(config.QUERY_EMBEDDINGS_PATH)
    passage_embeddings = np.load(config.PASSAGE_EMBEDDINGS_PATH)

    with open(config.QUERY_ID_TO_IDX_PATH, 'r') as f:
        query_id_to_idx = json.load(f)

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
    parser = argparse.ArgumentParser(description='Generate embeddings for MS MARCO passages and queries')
    parser.add_argument('--skip-if-exists', action='store_true',
                        help='Skip generation if embeddings already exist')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing embeddings')
    parser.add_argument('--model-name', type=str, default=None,
                        help=f'SBERT model name (default: {config.SBERT_MODEL_NAME})')

    args = parser.parse_args()

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
    unique_qids, unique_pids = collect_ids_from_files()

    # Step 2: Load texts for these IDs
    unique_qids, unique_pids, qid_to_text, pid_to_text = load_texts(unique_qids, unique_pids)

    # Step 3: Generate embeddings
    query_embeddings, passage_embeddings, query_id_to_idx, passage_id_to_idx = generate_embeddings(
        qid_to_text, pid_to_text, args.model_name
    )

    # Step 4: Save everything
    save_embeddings_and_mappings(query_embeddings, passage_embeddings,
                                 query_id_to_idx, passage_id_to_idx)

    # Step 5: Verify
    verify_embeddings()

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("Embedding generation complete!")


if __name__ == "__main__":
    main()