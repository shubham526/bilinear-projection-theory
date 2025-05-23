import os
import numpy as np
import json
import ir_datasets
import requests
import tarfile
import io
from tqdm import tqdm
import argparse
import time
import torch
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


def generate_chunked_embeddings(doc_text, tokenizer, model, max_length=512, stride=256,
                                device="cuda", pooling="mean", aggregation="hybrid"):
    """
    Generate embeddings for a long document by chunking it and then aggregating the embeddings.

    Args:
        doc_text: The full document text
        tokenizer: HuggingFace tokenizer
        model: The embedding model
        max_length: Maximum chunk length
        stride: Overlap between chunks
        device: Device to run the model on
        pooling: Pooling strategy ('mean' or 'cls')
        aggregation: Method to aggregate chunk embeddings

    Returns:
        A single embedding vector representing the document
    """
    # Tokenize the entire document
    tokens = tokenizer.tokenize(doc_text)

    # If document is short enough, no need for chunking
    if len(tokens) <= max_length - 2:  # Account for [CLS] and [SEP]
        inputs = tokenizer(doc_text, padding=True, truncation=True,
                           return_tensors="pt", max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                torch.sum(input_mask_expanded, 1), min=1e-9)
        else:
            # CLS pooling
            embedding = outputs.last_hidden_state[:, 0]

        return embedding.cpu().numpy()[0]

    # For long documents, create overlapping chunks
    chunk_embeddings = []

    # Process document in chunks with overlap
    for i in range(0, len(tokens), stride):
        # Extract chunk tokens
        chunk_tokens = tokens[i:i + max_length - 2]  # Account for [CLS] and [SEP]
        if not chunk_tokens:
            continue

        # Convert tokens back to text
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        # Embed the chunk
        inputs = tokenizer(chunk_text, padding=True, truncation=True,
                           return_tensors="pt", max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            chunk_embed = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                torch.sum(input_mask_expanded, 1), min=1e-9)
        else:
            # CLS pooling
            chunk_embed = outputs.last_hidden_state[:, 0]

        chunk_embeddings.append(chunk_embed.cpu().numpy()[0])

    # Return aggregated embeddings
    return aggregate_chunk_embeddings(chunk_embeddings, aggregation)


def aggregate_chunk_embeddings(chunks, method="hybrid"):
    """
    Aggregate chunk embeddings using various strategies.

    Args:
        chunks: List of chunk embeddings
        method: Aggregation method

    Returns:
        A single embedding vector
    """
    if not chunks:
        # Return zeros array matching embedding dimension (usually 768)
        if hasattr(chunks, 'shape') and len(chunks.shape) > 0:
            return np.zeros(chunks.shape[1])
        return np.zeros(768)

    # Different aggregation strategies
    if method == "mean":
        # Simple mean pooling
        return np.mean(chunks, axis=0)

    elif method == "max":
        # Max pooling across chunks
        return np.max(chunks, axis=0)

    elif method == "position":
        # Weight chunks by position (early chunks get higher weight)
        decay_factor = 0.8
        weights = np.array([decay_factor ** i for i in range(len(chunks))])
        weights = weights / weights.sum()  # Normalize
        return np.sum([w * emb for w, emb in zip(weights, chunks)], axis=0)

    elif method == "importance":
        # Weight chunks by their L2 norm (proxy for information density)
        norms = np.array([np.linalg.norm(emb) for emb in chunks])
        if np.sum(norms) == 0:
            return np.mean(chunks, axis=0)  # Fallback
        weights = norms / np.sum(norms)
        return np.sum([w * emb for w, emb in zip(weights, chunks)], axis=0)

    elif method == "first_chunk":
        # Just use the first chunk
        return chunks[0]

    elif method == "hybrid":
        # Combine multiple methods
        methods = []
        # Mean embedding
        methods.append(np.mean(chunks, axis=0))
        # Max embedding
        methods.append(np.max(chunks, axis=0))
        # First chunk (often contains title/abstract)
        methods.append(chunks[0])
        # Last chunk (often contains conclusion)
        methods.append(chunks[-1])
        # Return average of all methods
        return np.mean(methods, axis=0)

    else:
        # Default to mean pooling
        return np.mean(chunks, axis=0)


def generate_embeddings(qid_to_text, pid_to_text, model_name=None, device=None, dataset_name=None):
    """
    Generate embeddings for all queries and passages using the specified model.
    Supports SBERT, HuggingFace transformers, and more.
    """
    if model_name is None:
        model_name = getattr(config, 'EMBEDDING_MODEL_NAME', config.SBERT_MODEL_NAME)

    if device is None:
        device = config.DEVICE

    print(f"Generating embeddings using model: {model_name}")
    print(f"Using device: {device}")

    # Sort IDs for consistent indexing
    sorted_qids = sorted(list(qid_to_text.keys()))
    sorted_pids = sorted(list(pid_to_text.keys()))

    # Create mappings
    query_id_to_idx = {qid: i for i, qid in enumerate(sorted_qids)}
    passage_id_to_idx = {pid: i for i, pid in enumerate(sorted_pids)}

    # Extract text lists
    query_texts_list = [qid_to_text[qid] for qid in sorted_qids]
    passage_texts_list = [pid_to_text[pid] for pid in sorted_pids]

    # Get chunking parameters for ROBUST documents
    chunk_size = getattr(config, 'ROBUST_CHUNK_SIZE', 512)
    chunk_stride = getattr(config, 'ROBUST_CHUNK_STRIDE', 256)
    chunk_aggregation = getattr(config, 'ROBUST_CHUNK_AGGREGATION', 'hybrid')
    is_robust = hasattr(config, 'ROBUST_QUERIES_FILE')

    # APPROACH 1: SENTENCE TRANSFORMERS (SBERT)
    if "sentence-transformers" in model_name or any(
            name in model_name for name in ["all-mpnet", "all-MiniLM", "all-distilroberta"]):
        print(f"Using Sentence Transformers approach with model: {model_name}")
        from sentence_transformers import SentenceTransformer

        sbert_model = SentenceTransformer(model_name, device=device)

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

        # Special handling for ROBUST documents
        if is_robust and dataset_name == "robust":
            print(f"Using chunking for ROBUST documents with {chunk_aggregation} aggregation...")

            # We need HuggingFace tokenizer/model for chunking with SBERT
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_name.replace('sentence-transformers/', ''))

            # Initialize array for document embeddings
            passage_embeddings_np = np.zeros((len(passage_texts_list), sbert_model.get_sentence_embedding_dimension()))

            for i, passage_text in enumerate(tqdm(passage_texts_list, desc="Chunking ROBUST documents")):
                if i < 3:  # Log a few examples
                    print(f"Document {i}: Length={len(tokenizer.tokenize(passage_text))} tokens")

                # Generate chunks and encode with SBERT
                chunks = []
                # Tokenize the document
                tokens = tokenizer.tokenize(passage_text)

                # If short enough, encode directly
                if len(tokens) <= chunk_size - 2:  # Account for special tokens
                    passage_embeddings_np[i] = sbert_model.encode(
                        passage_text,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    continue

                # Create chunks with overlap
                for j in range(0, len(tokens), chunk_stride):
                    chunk_tokens = tokens[j:j + chunk_size - 2]
                    if not chunk_tokens:
                        continue

                    chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                    chunk_embedding = sbert_model.encode(
                        chunk_text,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    chunks.append(chunk_embedding)

                # Aggregate chunks
                if chunks:
                    passage_embeddings_np[i] = aggregate_chunk_embeddings(chunks, chunk_aggregation)
                else:
                    # Fallback: encode with truncation
                    passage_embeddings_np[i] = sbert_model.encode(
                        passage_text,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
        else:
            # Standard encoding for non-ROBUST documents
            passage_embeddings_np = sbert_model.encode(
                passage_texts_list,
                batch_size=128,  # Smaller batch size for longer passages
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False  # Keep raw embeddings
            )

    # APPROACH 2: STANDARD HUGGINGFACE TRANSFORMERS
    else:
        print(f"Using HuggingFace Transformers approach with model: {model_name}")
        from transformers import AutoTokenizer, AutoModel

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        # Mean Pooling function
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        # CLS Pooling function
        def cls_pooling(model_output):
            return model_output.last_hidden_state[:, 0]

        # Choose pooling strategy - could be a parameter or in config
        pooling_strategy = "mean"  # or "cls"
        print(f"Using {pooling_strategy} pooling for transformer embeddings")

        # Helper function to embed batches of text
        def encode_batch(texts, batch_size=32, pooling_strategy="mean"):
            all_embeddings = []

            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded_input = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(device)

                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)

                # Pool embeddings
                if pooling_strategy == "mean":
                    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                elif pooling_strategy == "cls":
                    embeddings = cls_pooling(model_output)
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

                # Move to CPU and convert to numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

            return np.vstack(all_embeddings)

        # Generate query embeddings
        print(f"Encoding {len(query_texts_list)} query texts...")
        query_embeddings_np = encode_batch(
            query_texts_list,
            batch_size=64,  # Adjust based on your GPU memory
            pooling_strategy=pooling_strategy
        )

        # Generate passage embeddings
        print(f"Encoding {len(passage_texts_list)} document/passage texts...")

        # Special handling for ROBUST documents
        if is_robust and dataset_name == "robust":
            print(f"Using chunking for ROBUST documents with {chunk_aggregation} aggregation...")

            # Initialize array for document embeddings
            passage_embeddings_np = np.zeros((len(passage_texts_list), model.config.hidden_size))

            for i, passage_text in enumerate(tqdm(passage_texts_list, desc="Chunking ROBUST documents")):
                if i < 3:  # Log a few examples
                    print(f"Document {i}: Length={len(tokenizer.tokenize(passage_text))} tokens")

                # Use the chunking function
                passage_embeddings_np[i] = generate_chunked_embeddings(
                    passage_text,
                    tokenizer,
                    model,
                    max_length=chunk_size,
                    stride=chunk_stride,
                    device=device,
                    pooling=pooling_strategy,
                    aggregation=chunk_aggregation
                )
        else:
            # Standard encoding for non-ROBUST documents
            passage_embeddings_np = encode_batch(
                passage_texts_list,
                batch_size=32,  # Smaller for passages as they are typically longer
                pooling_strategy=pooling_strategy
            )

    print(f"Query embeddings shape: {query_embeddings_np.shape}")
    print(f"Document/passage embeddings shape: {passage_embeddings_np.shape}")

    return query_embeddings_np, passage_embeddings_np, query_id_to_idx, passage_id_to_idx


def save_embeddings_and_mappings(query_embeddings_np, passage_embeddings_np,
                                 query_id_to_idx, passage_id_to_idx, model_name, dataset_name):
    """
    Save embeddings and mapping files using direct paths in EMBEDDING_DIR.
    """
    print("Saving embeddings and mappings...")

    # Create directory if it doesn't exist
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)

    # Use simple, direct filenames in the EMBEDDING_DIR (no config path lookup)
    query_embeddings_path = os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_embeddings.npy")
    passage_embeddings_path = os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_passage_embeddings.npy")
    query_id_to_idx_path = os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_id_to_idx.json")
    passage_id_to_idx_path = os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_passage_id_to_idx.json")

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
    """Main function to generate embeddings for IR datasets."""
    parser = argparse.ArgumentParser(
        description='Generate embeddings for IR datasets using ir_datasets')
    parser.add_argument('--dataset', type=str, choices=['msmarco', 'car', 'robust'], required=True,
                        help='Dataset to use: msmarco, car, or robust')
    parser.add_argument('--model-name', type=str, default=None,
                        help=f'Model name (can be SBERT or HuggingFace model)')
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
    parser.add_argument('--pooling', type=str, choices=['mean', 'cls'], default='mean',
                        help='Pooling strategy for HuggingFace models (default: mean)')
    parser.add_argument('--use-chunking', action='store_true',
                        help='Use chunking for long documents (ROBUST dataset)')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Maximum chunk size in tokens (default: 512)')
    parser.add_argument('--chunk-stride', type=int, default=256,
                        help='Stride between chunks (default: 256)')
    parser.add_argument('--chunk-aggregation', type=str,
                        choices=['mean', 'max', 'position', 'importance', 'first_chunk', 'hybrid'],
                        default='hybrid',
                        help='Method to aggregate chunk embeddings (default: hybrid)')
    parser.add_argument('--embedding-dir', type=str, default=None,
                        help='Directory to store embeddings (overrides config.EMBEDDING_DIR)')

    args = parser.parse_args()

    # Set dataset name and model name
    dataset_name = args.dataset
    model_name = args.model_name if args.model_name else getattr(config, 'EMBEDDING_MODEL_NAME', config.SBERT_MODEL_NAME)
    device = args.device

    # Override embedding directory if specified
    if args.embedding_dir:
        original_embedding_dir = config.EMBEDDING_DIR
        config.EMBEDDING_DIR = args.embedding_dir
        os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
        print(f"Created embedding directory: {config.EMBEDDING_DIR}")



        # Also update all dependent paths
        for dataset_prefix in ['', 'CAR_', 'ROBUST_']:
            if hasattr(config, f"{dataset_prefix}QUERY_EMBEDDINGS_PATH"):
                setattr(config, f"{dataset_prefix}QUERY_EMBEDDINGS_PATH",
                        getattr(config, f"{dataset_prefix}QUERY_EMBEDDINGS_PATH").replace(original_embedding_dir,
                                                                                          args.embedding_dir))
            if hasattr(config, f"{dataset_prefix}PASSAGE_EMBEDDINGS_PATH"):
                setattr(config, f"{dataset_prefix}PASSAGE_EMBEDDINGS_PATH",
                        getattr(config, f"{dataset_prefix}PASSAGE_EMBEDDINGS_PATH").replace(original_embedding_dir,
                                                                                            args.embedding_dir))
            if hasattr(config, f"{dataset_prefix}QUERY_ID_TO_IDX_PATH"):
                setattr(config, f"{dataset_prefix}QUERY_ID_TO_IDX_PATH",
                        getattr(config, f"{dataset_prefix}QUERY_ID_TO_IDX_PATH").replace(original_embedding_dir,
                                                                                         args.embedding_dir))
            if hasattr(config, f"{dataset_prefix}PASSAGE_ID_TO_IDX_PATH"):
                setattr(config, f"{dataset_prefix}PASSAGE_ID_TO_IDX_PATH",
                        getattr(config, f"{dataset_prefix}PASSAGE_ID_TO_IDX_PATH").replace(original_embedding_dir,
                                                                                           args.embedding_dir))

        print(f"Embedding directory set to: {config.EMBEDDING_DIR} (from command-line argument)")

    # Handle verify-only mode
    if args.verify_only:
        verify_embeddings(dataset_name)
        return

    # Get config values
    config_prefix = dataset_name.upper()
    query_embeddings_path = getattr(config, f"{config_prefix}_QUERY_EMBEDDINGS_PATH",
                                    os.path.join(config.EMBEDDING_DIR, f"{dataset_name}_query_embeddings.npy"))

    # Skip if embeddings already exist and --skip-if-exists flag is set
    if args.skip_if_exists and os.path.exists(query_embeddings_path):
        print(f"Embeddings for {dataset_name} already exist at {query_embeddings_path}")
        print("Skipping generation. Use --verify-only to check existing embeddings.")
        return

    # Override config file paths if provided via command line
    if args.queries_file and (dataset_name == "car" or dataset_name == "robust"):
        if dataset_name == "car":
            config.CAR_QUERIES_FILE = args.queries_file
        else:
            config.ROBUST_QUERIES_FILE = args.queries_file

    if args.qrels_file and (dataset_name == "car" or dataset_name == "robust"):
        if dataset_name == "car":
            config.CAR_QRELS_FILE = args.qrels_file
        else:
            config.ROBUST_QRELS_FILE = args.qrels_file

    if args.run_file and (dataset_name == "car" or dataset_name == "robust"):
        if dataset_name == "car":
            config.CAR_RUN_FILE = args.run_file
        else:
            config.ROBUST_RUN_FILE = args.run_file

    # Add chunking parameters to config for ROBUST documents
    if dataset_name == "robust" or args.use_chunking:
        config.ROBUST_USE_CHUNKING = args.use_chunking
        config.ROBUST_CHUNK_SIZE = args.chunk_size
        config.ROBUST_CHUNK_STRIDE = args.chunk_stride
        config.ROBUST_CHUNK_AGGREGATION = args.chunk_aggregation
        print(f"Chunking enabled: size={args.chunk_size}, stride={args.chunk_stride}, "
              f"aggregation={args.chunk_aggregation}")

    # Load dataset parts
    dataset_parts = get_dataset_parts(dataset_name)

    # Collect unique IDs
    unique_qids, unique_pids = collect_unique_ids(dataset_name, dataset_parts)

    # Apply limits if specified
    if args.limit_queries and len(unique_qids) > args.limit_queries:
        print(f"Limiting to {args.limit_queries} queries (original: {len(unique_qids)})")
        unique_qids = set(list(unique_qids)[:args.limit_queries])

    if args.limit_passages and len(unique_pids) > args.limit_passages:
        print(f"Limiting to {args.limit_passages} passages (original: {len(unique_pids)})")
        unique_pids = set(list(unique_pids)[:args.limit_passages])

    # Load texts
    qid_to_text, pid_to_text = load_texts(unique_qids, unique_pids, dataset_name, dataset_parts)

    # Generate embeddings
    query_embeddings_np, passage_embeddings_np, query_id_to_idx, passage_id_to_idx = generate_embeddings(
        qid_to_text, pid_to_text, model_name, device, dataset_name
    )

    # Save embeddings and mappings
    save_embeddings_and_mappings(
        query_embeddings_np, passage_embeddings_np,
        query_id_to_idx, passage_id_to_idx,
        model_name, dataset_name
    )

    # Verify embeddings
    verify_embeddings(dataset_name)

if __name__ == "__main__":
    main()