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


def read_trec_run_file(run_file_path):
    """
    Read a TREC-style run file and extract query IDs and document IDs.
    Format: qid Q0 docid rank score run_name
    Returns a dictionary mapping query IDs to lists of document IDs.
    """
    if not os.path.exists(run_file_path):
        print(f"Error: Run file not found at {run_file_path}")
        return {}

    query_to_docs = {}

    print(f"Reading TREC run file: {run_file_path}")
    with open(run_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Run file entries"):
            try:
                parts = line.strip().split()
                if len(parts) >= 6:  # TREC format: qid Q0 docid rank score run_name
                    qid = parts[0]
                    docid = parts[2]

                    if qid not in query_to_docs:
                        query_to_docs[qid] = []

                    query_to_docs[qid].append(docid)
            except Exception as e:
                print(f"Error parsing line in run file: {e}")
                continue

    print(
        f"Read {len(query_to_docs)} queries and {sum(len(docs) for docs in query_to_docs.values())} document references")
    return query_to_docs


def get_dataset_parts(dataset_name):
    """
    Get the appropriate dataset parts based on the dataset name.
    Returns a dictionary with dataset parts.
    """
    dataset_parts = {}

    # MS MARCO
    if dataset_name == "msmarco-passage" or dataset_name == "msmarco":
        dataset_parts["base"] = ir_datasets.load("msmarco-passage")
        dataset_parts["train"] = ir_datasets.load("msmarco-passage/train/triples-small")
        dataset_parts["dev"] = ir_datasets.load("msmarco-passage/dev/small")

    # TREC CAR
    elif dataset_name == "car":
        # For TREC CAR, only use v2.0 for documents
        dataset_parts["base"] = ir_datasets.load("car/v2.0")
        # We'll read queries/qrels from files provided in config.py

    # TREC ROBUST 2004
    elif dataset_name == "robust":
        dataset_parts["base"] = ir_datasets.load("disks45/nocr/trec-robust-2004")
        # We'll read queries/qrels from files provided in config.py

    else:
        try:
            # Try to load as a single dataset
            dataset_parts["base"] = ir_datasets.load(dataset_name)
        except:
            print(f"Error: Dataset {dataset_name} not recognized or not supported.")
            exit(1)

    return dataset_parts


def collect_unique_ids(dataset_name, dataset_parts):
    """
    Collect all unique query IDs and passage/document IDs that we need embeddings for.
    """
    print("Collecting unique query and passage/document IDs...")

    unique_qids = set()
    unique_pids = set()

    # Process based on dataset type
    # MS MARCO
    if dataset_name == "msmarco-passage" or dataset_name == "msmarco":
        # From training triples
        print("Reading training triples from ir_datasets...")
        train_dataset = dataset_parts["train"]

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
        dev_dataset = dataset_parts["dev"]

        print("Getting query IDs from dev dataset...")
        for query in tqdm(dev_dataset.queries_iter(), desc="Dev queries"):
            unique_qids.add(query.query_id)

        print("Getting passage IDs from dev qrels...")
        for qrel in tqdm(dev_dataset.qrels_iter(), desc="Dev qrels"):
            unique_pids.add(qrel.doc_id)

        # Use scoreddocs_iter for top candidates in MS MARCO
        if hasattr(dev_dataset, 'scoreddocs_iter'):
            print("Getting passage IDs from dev scoreddocs...")
            for scoreddoc in tqdm(dev_dataset.scoreddocs_iter(), desc="Dev scoreddocs"):
                unique_qids.add(scoreddoc.query_id)
                unique_pids.add(scoreddoc.doc_id)

    # TREC CAR
    elif dataset_name == "car":
        # Read queries from file
        queries_file = config.CAR_QUERIES_FILE
        if os.path.exists(queries_file):
            print(f"Reading queries from {queries_file}...")
            try:
                with open(queries_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Reading queries"):
                        # Assuming format: qid\ttitle/text
                        parts = line.strip().split('\t')
                        if len(parts) >= 1:
                            qid = parts[0]
                            unique_qids.add(qid)
            except Exception as e:
                print(f"Error reading queries file: {e}")

        # Read qrels from file
        qrels_file = config.CAR_QRELS_FILE
        if os.path.exists(qrels_file):
            print(f"Reading qrels from {qrels_file}...")
            try:
                with open(qrels_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Reading qrels"):
                        # Assuming format: qid 0 docid rel
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            qid = parts[0]
                            docid = parts[2]
                            unique_qids.add(qid)
                            unique_pids.add(docid)
            except Exception as e:
                print(f"Error reading qrels file: {e}")

        # Read run file
        run_file = config.CAR_RUN_FILE
        if os.path.exists(run_file):
            print(f"Reading run file from {run_file}...")
            query_to_docs = read_trec_run_file(run_file)

            # Add query IDs and document IDs from run file
            for qid, doc_ids in query_to_docs.items():
                unique_qids.add(qid)
                unique_pids.update(doc_ids)

    # TREC ROBUST 2004
    elif dataset_name == "robust":
        # Read queries from file
        queries_file = config.ROBUST_QUERIES_FILE
        if os.path.exists(queries_file):
            print(f"Reading queries from {queries_file}...")
            try:
                with open(queries_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Reading queries"):
                        # Assuming format: qid\ttitle/text
                        parts = line.strip().split('\t')
                        if len(parts) >= 1:
                            qid = parts[0]
                            unique_qids.add(qid)
            except Exception as e:
                print(f"Error reading queries file: {e}")

        # Read qrels from file
        qrels_file = config.ROBUST_QRELS_FILE
        if os.path.exists(qrels_file):
            print(f"Reading qrels from {qrels_file}...")
            try:
                with open(qrels_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Reading qrels"):
                        # Assuming format: qid 0 docid rel
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            qid = parts[0]
                            docid = parts[2]
                            unique_qids.add(qid)
                            unique_pids.add(docid)
            except Exception as e:
                print(f"Error reading qrels file: {e}")

        # Read run file
        run_file = config.ROBUST_RUN_FILE
        if os.path.exists(run_file):
            print(f"Reading run file from {run_file}...")
            query_to_docs = read_trec_run_file(run_file)

            # Add query IDs and document IDs from run file
            for qid, doc_ids in query_to_docs.items():
                unique_qids.add(qid)
                unique_pids.update(doc_ids)

    print(f"Found {len(unique_qids)} unique query IDs")
    print(f"Found {len(unique_pids)} unique document/passage IDs")

    return unique_qids, unique_pids


def extract_query_text(query, dataset_name):
    """
    Extract the text field from a query based on dataset type.
    """
    if dataset_name == "msmarco-passage" or dataset_name == "msmarco":
        return query.text
    elif dataset_name == "car":
        # For TREC CAR, combine title and headings for query text
        if hasattr(query, 'title') and hasattr(query, 'headings'):
            return f"{query.title} {query.headings}"
        elif hasattr(query, 'text'):
            return query.text
    elif dataset_name == "robust":
        # For TREC ROBUST, use title and optionally description
        if hasattr(query, 'title'):
            title = query.title or ""
            desc = query.description if hasattr(query, 'description') else ""
            return f"{title} {desc}".strip()

    # Fall back to text attribute if it exists, otherwise try title
    if hasattr(query, 'text'):
        return query.text
    elif hasattr(query, 'title'):
        return query.title
    else:
        # Try to find any text-like attribute
        for attr in ['query', 'content', 'raw']:
            if hasattr(query, attr):
                return getattr(query, attr)

        # Last resort: convert the first string attribute to text
        for attr_name in dir(query):
            if attr_name.startswith('_'):
                continue
            attr_value = getattr(query, attr_name)
            if isinstance(attr_value, str) and attr_value:
                return attr_value

        # If all else fails
        return str(query)


def extract_doc_text(doc, dataset_name):
    """
    Extract the text field from a document based on dataset type.
    """
    if dataset_name == "msmarco-passage" or dataset_name == "msmarco":
        return doc.text
    elif dataset_name == "car":
        return doc.text
    elif dataset_name == "robust":
        # For TREC ROBUST, combine title and body
        if hasattr(doc, 'title') and hasattr(doc, 'body'):
            title = doc.title or ""
            body = doc.body or ""
            return f"{title} {body}".strip()
        elif hasattr(doc, 'body'):
            return doc.body
        elif hasattr(doc, 'text'):
            return doc.text

    # Fall back to text attribute if it exists
    if hasattr(doc, 'text'):
        return doc.text
    elif hasattr(doc, 'body'):
        return doc.body
    elif hasattr(doc, 'content'):
        return doc.content
    else:
        # Try to find any text-like attribute
        for attr in ['raw', 'document', 'passage']:
            if hasattr(doc, attr):
                return getattr(doc, attr)

        # Last resort: convert the first string attribute to text
        for attr_name in dir(doc):
            if attr_name.startswith('_'):
                continue
            attr_value = getattr(doc, attr_name)
            if isinstance(attr_value, str) and attr_value:
                return attr_value

        # If all else fails
        return str(doc)


def load_queries_from_file(file_path, query_ids=None):
    """
    Load query texts from a file.
    Expected format: qid\ttext

    Args:
        file_path: Path to queries file
        query_ids: Optional set of query IDs to filter by

    Returns:
        dict: Mapping from query IDs to query text
    """
    qid_to_text = {}

    if not os.path.exists(file_path):
        print(f"Warning: Queries file not found: {file_path}")
        return qid_to_text

    print(f"Loading queries from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading queries"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    text = parts[1]

                    # Filter by query_ids if provided
                    if query_ids is None or qid in query_ids:
                        qid_to_text[qid] = text
    except Exception as e:
        print(f"Error reading queries file: {e}")

    return qid_to_text


def load_texts(unique_qids, unique_pids, dataset_name, dataset_parts):
    """
    Load the actual text for the collected IDs using ir_datasets.
    """
    print("Loading query and passage/document texts...")

    qid_to_text = {}
    pid_to_text = {}

    # Load query texts first
    if dataset_name == "msmarco-passage" or dataset_name == "msmarco":
        # MS MARCO specific loading
        train_dataset = dataset_parts["train"]
        dev_dataset = dataset_parts["dev"]

        # Load query texts from train dataset
        print("Loading query texts from train dataset...")
        for query in tqdm(train_dataset.queries_iter(), desc="Train queries"):
            if query.query_id in unique_qids:
                qid_to_text[query.query_id] = extract_query_text(query, dataset_name)

        # Load query texts from dev dataset
        print("Loading query texts from dev dataset...")
        for query in tqdm(dev_dataset.queries_iter(), desc="Dev queries"):
            if query.query_id in unique_qids:
                qid_to_text[query.query_id] = extract_query_text(query, dataset_name)

    elif dataset_name == "car":
        # Load query texts from file
        queries_file = config.CAR_QUERIES_FILE
        if os.path.exists(queries_file):
            queries_from_file = load_queries_from_file(queries_file, unique_qids)
            qid_to_text.update(queries_from_file)

    elif dataset_name == "robust":
        # Load query texts from file
        queries_file = config.ROBUST_QUERIES_FILE
        if os.path.exists(queries_file):
            queries_from_file = load_queries_from_file(queries_file, unique_qids)
            qid_to_text.update(queries_from_file)

        # As a backup, try to load from ir_datasets if file loading didn't get all queries
        if len(qid_to_text) < len(unique_qids):
            missing_qids = unique_qids - set(qid_to_text.keys())
            print(f"Trying to load {len(missing_qids)} missing queries from ir_datasets...")

            # Load from ir_datasets
            main_dataset = dataset_parts.get("base")
            if hasattr(main_dataset, 'queries_iter'):
                for query in tqdm(main_dataset.queries_iter(), desc="Loading queries from ir_datasets"):
                    if query.query_id in missing_qids:
                        qid_to_text[query.query_id] = extract_query_text(query, dataset_name)

    # Now load document texts
    # Determine the appropriate dataset for documents
    base_dataset = dataset_parts.get("base")

    if base_dataset is None or not hasattr(base_dataset, 'docs_iter'):
        print(f"Error: No document iterator found for dataset {dataset_name}")
        exit(1)

    # Load document/passage texts
    print(f"Fetching texts for {len(unique_pids)} documents/passages. This may take a while...")

    # Process documents in batches to show progress
    try:
        total_docs = base_dataset.docs_count()
        print(f"Total documents in dataset: {total_docs}")
    except:
        # If docs_count() is not available
        print("Document count not available, using progress without total")
        total_docs = None

    batch_size = 100000  # Process in batches to show progress

    # Create a set for faster lookups
    unique_pids_set = set(unique_pids)
    remaining_pids = set(unique_pids)

    # Process documents and update progress every batch
    processed = 0
    doc_iter = base_dataset.docs_iter()

    with tqdm(total=total_docs, desc="Processing documents") as pbar:
        try:
            while remaining_pids and (total_docs is None or processed < total_docs):
                batch_processed = 0
                for _ in range(batch_size):
                    if not remaining_pids:
                        break

                    try:
                        doc = next(doc_iter)
                        processed += 1
                        batch_processed += 1

                        # Check if this doc is in our target set
                        if doc.doc_id in unique_pids_set:
                            pid_to_text[doc.doc_id] = extract_doc_text(doc, dataset_name)
                            remaining_pids.discard(doc.doc_id)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error processing document: {e}")
                        continue

                # Update progress bar
                if total_docs:
                    pbar.update(batch_processed)
                else:
                    pbar.update(0)  # Just to refresh the display

                pbar.set_postfix({"found": len(pid_to_text), "remaining": len(remaining_pids), "processed": processed})

                if not remaining_pids or (total_docs and processed >= total_docs):
                    break

        except Exception as e:
            print(f"Error during document processing: {e}")
            # Continue with what we've got

    print(f"Loaded text for {len(qid_to_text)} queries")
    print(f"Loaded text for {len(pid_to_text)} documents/passages")

    # Log warnings about missing texts
    missing_qids = set(unique_qids) - set(qid_to_text.keys())
    missing_pids = set(unique_pids) - set(pid_to_text.keys())

    if missing_qids:
        print(f"Warning: Couldn't find text for {len(missing_qids)} query IDs")
        if len(missing_qids) < 10:
            print(f"Missing query IDs: {missing_qids}")

    if missing_pids:
        print(f"Warning: Couldn't find text for {len(missing_pids)} document/passage IDs")
        if len(missing_pids) < 10:
            print(f"Missing document IDs: {missing_pids}")
        else:
            print(f"First 10 missing document IDs: {list(missing_pids)[:10]}")

    return qid_to_text, pid_to_text


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
    print(f"Encoding {len(passage_texts_list)} document/passage texts...")
    passage_embeddings_np = sbert_model.encode(
        passage_texts_list,
        batch_size=128,  # Smaller batch size for longer passages
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Keep raw embeddings
    )

    print(f"Query embeddings shape: {query_embeddings_np.shape}")
    print(f"Document/passage embeddings shape: {passage_embeddings_np.shape}")

    return query_embeddings_np, passage_embeddings_np, query_id_to_idx, passage_id_to_idx


def save_embeddings_and_mappings(query_embeddings_np, passage_embeddings_np,
                                 query_id_to_idx, passage_id_to_idx, model_name, dataset_name):
    """
    Save embeddings and mapping files using paths from config.
    """
    print("Saving embeddings and mappings...")

    # Create directory if it doesn't exist
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)

    # Format dataset name for config attributes
    config_prefix = dataset_name.upper()

    # Use dataset-specific filenames if provided in config
    query_embeddings_path = getattr(config, f"{config_prefix}_QUERY_EMBEDDINGS_PATH",
                                    os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_embeddings.npy"))
    passage_embeddings_path = getattr(config, f"{config_prefix}_PASSAGE_EMBEDDINGS_PATH",
                                      os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_passage_embeddings.npy"))
    query_id_to_idx_path = getattr(config, f"{config_prefix}_QUERY_ID_TO_IDX_PATH",
                                   os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_id_to_idx.json"))
    passage_id_to_idx_path = getattr(config, f"{config_prefix}_PASSAGE_ID_TO_IDX_PATH",
                                     os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_passage_id_to_idx.json"))

    # Save embeddings
    print(f"Saving query embeddings to {query_embeddings_path}...")
    np.save(query_embeddings_path, query_embeddings_np)

    print(f"Saving document/passage embeddings to {passage_embeddings_path}...")
    np.save(passage_embeddings_path, passage_embeddings_np)

    # Save ID mappings
    print(f"Saving query ID to index mapping to {query_id_to_idx_path}...")
    with open(query_id_to_idx_path, 'w') as f:
        json.dump(query_id_to_idx, f, indent=2)

    print(f"Saving document/passage ID to index mapping to {passage_id_to_idx_path}...")
    with open(passage_id_to_idx_path, 'w') as f:
        json.dump(passage_id_to_idx, f, indent=2)

    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'num_queries': len(query_id_to_idx),
        'num_documents': len(passage_id_to_idx),
        'embedding_dim': query_embeddings_np.shape[1],
        'model_name': model_name,
        'generated_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = os.path.join(config.EMBEDDING_DIR, f'{dataset_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("All files saved successfully!")

    # Return paths for verification
    return query_embeddings_path, passage_embeddings_path, query_id_to_idx_path, passage_id_to_idx_path


def verify_embeddings(dataset_name=None):
    """
    Quick verification of saved embeddings.
    """
    print("Verifying saved embeddings...")

    # Format dataset name for config attributes if provided
    if dataset_name:
        config_prefix = dataset_name.upper()
        query_embeddings_path = getattr(config, f"{config_prefix}_QUERY_EMBEDDINGS_PATH",
                                        os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_embeddings.npy"))
        passage_embeddings_path = getattr(config, f"{config_prefix}_PASSAGE_EMBEDDINGS_PATH",
                                          os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_passage_embeddings.npy"))
        query_id_to_idx_path = getattr(config, f"{config_prefix}_QUERY_ID_TO_IDX_PATH",
                                       os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_id_to_idx.json"))
        passage_id_to_idx_path = getattr(config, f"{config_prefix}_PASSAGE_ID_TO_IDX_PATH",
                                         os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_passage_id_to_idx.json"))
    else:
        # Use default paths
        query_embeddings_path = config.QUERY_EMBEDDINGS_PATH
        passage_embeddings_path = config.PASSAGE_EMBEDDINGS_PATH
        query_id_to_idx_path = config.QUERY_ID_TO_IDX_PATH
        passage_id_to_idx_path = config.PASSAGE_ID_TO_IDX_PATH

    # Check if files exist
    files_to_check = [
        query_embeddings_path,
        passage_embeddings_path,
        query_id_to_idx_path,
        passage_id_to_idx_path
    ]

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found")
            return False

    # Load and check embeddings
    print(f"Loading saved query embeddings from {query_embeddings_path}...")
    query_embeddings = np.load(query_embeddings_path)

    print(f"Loading saved document/passage embeddings from {passage_embeddings_path}...")
    passage_embeddings = np.load(passage_embeddings_path)

    print(f"Loading query ID to index mapping from {query_id_to_idx_path}...")
    with open(query_id_to_idx_path, 'r') as f:
        query_id_to_idx = json.load(f)

    print(f"Loading document/passage ID to index mapping from {passage_id_to_idx_path}...")
    with open(passage_id_to_idx_path, 'r') as f:
        passage_id_to_idx = json.load(f)

    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Document/passage embeddings shape: {passage_embeddings.shape}")
    print(f"Number of query IDs: {len(query_id_to_idx)}")
    print(f"Number of document/passage IDs: {len(passage_id_to_idx)}")

    # Verify consistency
    assert query_embeddings.shape[0] == len(query_id_to_idx), "Query embedding count mismatch"
    assert passage_embeddings.shape[0] == len(passage_id_to_idx), "Document/passage embedding count mismatch"

    print("Verification successful!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for IR datasets using ir_datasets')
    parser.add_argument('--dataset', type=str, choices=['msmarco', 'car', 'robust'], required=True,
                        help='Dataset to use: msmarco, car, or robust')
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
    parser.add_argument('--queries-file', type=str, default=None,
                        help='Path to queries file (for car and robust, overrides config)')
    parser.add_argument('--qrels-file', type=str, default=None,
                        help='Path to qrels file (for car and robust, overrides config)')
    parser.add_argument('--run-file', type=str, default=None,
                        help='Path to run file (for car and robust, overrides config)')

    args = parser.parse_args()

    # Set dataset name and model name
    dataset_name = args.dataset
    model_name = args.model_name if args.model_name else config.SBERT