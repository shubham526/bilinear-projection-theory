#!/usr/bin/env python
# evaluate.py
import pytrec_eval
import torch
from tqdm import tqdm
import numpy as np
import config
import os
import ir_datasets


def load_qrels(qrels_path=None, use_ir_datasets=True):
    """
    Load qrels file either from a file path or using ir_datasets.

    Args:
        qrels_path: Path to qrels file (only used if use_ir_datasets=False)
        use_ir_datasets: Whether to use ir_datasets or file path

    Returns:
        Dictionary in format {qid: {pid: relevance_score}}
    """
    qrels = {}

    if use_ir_datasets:
        print("Loading qrels from ir_datasets...")
        try:
            # Load dev dataset from ir_datasets
            dev_dataset = ir_datasets.load("msmarco-passage/dev/small")

            # Process qrels
            for qrel in tqdm(dev_dataset.qrels_iter(), desc="Loading qrels"):
                qid = qrel.query_id
                pid = qrel.doc_id
                relevance = qrel.relevance

                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][pid] = int(relevance)

            print(f"Loaded qrels for {len(qrels)} queries from ir_datasets")
            return qrels

        except Exception as e:
            print(f"Error loading qrels from ir_datasets: {e}")
            print("Falling back to file-based loading...")
            # Will continue with file-based loading below if ir_datasets fails

    # File-based loading (fallback or if requested)
    if not qrels_path:
        qrels_path = config.DEV_QRELS_PATH

    print(f"Loading qrels from file: {qrels_path}")

    try:
        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, pid, relevance = parts[:4]
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][pid] = int(relevance)
    except Exception as e:
        print(f"Error loading qrels from file: {e}")
        return {}

    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def evaluate_model_on_dev(model, query_embeddings, passage_embeddings,
                          qid_to_idx, pid_to_idx, dev_query_to_candidates,
                          run_file_path="run.dev.txt", use_ir_datasets=True):
    """
    Evaluate model on dev set by scoring candidates and creating a TREC run file.
    Then evaluates using pytrec_eval.

    Args:
        model: The model to evaluate
        query_embeddings: Query embeddings numpy array
        passage_embeddings: Passage embeddings numpy array
        qid_to_idx: Mapping from query IDs to embedding indices
        pid_to_idx: Mapping from passage IDs to embedding indices
        dev_query_to_candidates: Dictionary mapping queries to candidates
        run_file_path: Path to write run file
        use_ir_datasets: Whether to use ir_datasets for qrels loading
    """
    model.eval()

    # Ensure run_file_path directory exists
    os.makedirs(os.path.dirname(run_file_path), exist_ok=True)

    run = {}  # Will store run results for pytrec_eval

    with open(run_file_path, 'w') as f_run:
        with torch.no_grad():
            for qid, candidate_pids in tqdm(dev_query_to_candidates.items(), desc="Evaluating Dev Set"):
                if not candidate_pids:  # Skip if no candidates for this query
                    continue

                try:
                    q_embed_idx = qid_to_idx[qid]
                    q_embed_np = query_embeddings[q_embed_idx]
                    q_embed = torch.tensor(q_embed_np, dtype=torch.float).unsqueeze(0).to(config.DEVICE)  # (1, D)
                except KeyError:
                    print(f"Warning: Query ID {qid} not found in embedding mapping. Skipping.")
                    continue

                candidate_p_embeds_list = []
                valid_candidate_pids = []

                # Handle different formats of candidate_pids:
                # - If it's from the file-based loader, it might be a list of PIDs
                # - If it's from ir_datasets loader, it might be a list of tuples (pid, qidx, pidx)
                for item in candidate_pids:
                    pid = item[0] if isinstance(item, tuple) else item
                    try:
                        p_embed_idx = pid_to_idx[pid]
                        candidate_p_embeds_list.append(passage_embeddings[p_embed_idx])
                        valid_candidate_pids.append(pid)
                    except KeyError:
                        # Skip passages not in our embedding mapping
                        continue

                if not candidate_p_embeds_list:
                    continue

                candidate_p_embeds = torch.tensor(np.array(candidate_p_embeds_list), dtype=torch.float).to(
                    config.DEVICE)  # (num_candidates, D)

                # Expand query embedding to match number of candidates for batch scoring
                q_embed_expanded = q_embed.expand(candidate_p_embeds.size(0), -1)  # (num_candidates, D)

                scores = model(q_embed_expanded, candidate_p_embeds)  # (num_candidates,)

                # Sort candidates by score (descending)
                sorted_indices = torch.argsort(scores, descending=True)

                # Prepare for pytrec_eval
                run[qid] = {}

                for rank_idx, original_candidate_idx in enumerate(sorted_indices):
                    passage_id = valid_candidate_pids[original_candidate_idx.item()]
                    score = scores[original_candidate_idx.item()].item()

                    # Store in run dictionary for pytrec_eval
                    run[qid][passage_id] = score

                    # Write in TREC run format: qid Q0 pid rank score run_name
                    f_run.write(f"{qid}\tQ0\t{passage_id}\t{rank_idx + 1}\t{score:.6f}\tBilinearModel\n")

    # Evaluate with pytrec_eval
    return evaluate_with_pytrec_eval(None, run, use_ir_datasets=use_ir_datasets)


def evaluate_with_pytrec_eval(qrels_path, run_dict, use_ir_datasets=True):
    """
    Evaluate using pytrec_eval

    Args:
        qrels_path: Path to qrels file (only used if use_ir_datasets=False)
        run_dict: Dictionary with run results
        use_ir_datasets: Whether to use ir_datasets for qrels loading

    Returns:
        (mrr_10, all_metrics_dict)
    """
    # Load qrels
    qrels = load_qrels(qrels_path, use_ir_datasets=use_ir_datasets)

    if not qrels:
        print("No qrels found. Cannot evaluate.")
        return 0.0, {}

    print(f"Evaluating {len(run_dict)} queries with pytrec_eval")

    # Use only metrics that are known to be supported
    metrics_to_evaluate = {
        'ndcg_cut_10', 'ndcg_cut_100',
        'recall_100', 'recall_1000',
        'map',
        'recip_rank'  # This is the standard MRR metric
    }

    # Create evaluator with metrics
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        metrics_to_evaluate
    )

    # Run evaluation
    results = evaluator.evaluate(run_dict)

    # Calculate average metrics across all queries
    aggregated = {}

    if results:
        # Get all metrics from first query to know what's available
        first_qid = next(iter(results))
        for metric in results[first_qid]:
            values = [results[qid][metric] for qid in results if qid in results]
            aggregated[metric] = np.mean(values) if values else 0.0

    # Add manual MRR@10 calculation
    mrr_10 = calculate_mrr_at_k(run_dict, qrels, k=10)
    aggregated['mrr_cut_10'] = mrr_10

    # Print formatted results
    print("\nEvaluation Results:")
    print("=" * 40)
    print(f"{'Metric':<15} {'Value':<10}")
    print("-" * 40)

    # Print metrics in a nice order
    print(f"{'MRR@10':<15} {aggregated.get('mrr_cut_10', 0):.4f}")
    print(f"{'MRR':<15} {aggregated.get('recip_rank', 0):.4f}")
    print(f"{'nDCG@10':<15} {aggregated.get('ndcg_cut_10', 0):.4f}")
    print(f"{'nDCG@100':<15} {aggregated.get('ndcg_cut_100', 0):.4f}")
    print(f"{'Recall@100':<15} {aggregated.get('recall_100', 0):.4f}")
    print(f"{'Recall@1000':<15} {aggregated.get('recall_1000', 0):.4f}")
    print(f"{'MAP':<15} {aggregated.get('map', 0):.4f}")
    print("=" * 40)

    # Return MRR@10 as primary metric and all metrics
    return mrr_10, aggregated


def calculate_mrr_at_k(run_dict, qrels, k=10):
    """
    Calculate Mean Reciprocal Rank at cutoff k manually.

    Args:
        run_dict: Dictionary of {qid: {pid: score}}
        qrels: Dictionary of {qid: {pid: relevance}}
        k: Cutoff rank

    Returns:
        MRR@k value
    """
    reciprocal_ranks = []

    for qid in qrels:
        if qid not in run_dict:
            continue

        # Get relevant document IDs for this query
        relevant_docs = {pid for pid, rel in qrels[qid].items() if rel > 0}
        if not relevant_docs:
            continue

        # Sort documents by score for this query
        sorted_docs = sorted(run_dict[qid].items(), key=lambda x: x[1], reverse=True)[:k]

        # Find the first relevant document's rank
        for rank, (pid, _) in enumerate(sorted_docs, start=1):
            if pid in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant docs in top k
            reciprocal_ranks.append(0.0)

    # Calculate mean
    if reciprocal_ranks:
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    else:
        return 0.0


def quick_eval_sample(model, query_embeddings, passage_embeddings,
                      qid_to_idx, pid_to_idx, dev_query_to_candidates,
                      sample_size=100, use_ir_datasets=True):
    """
    Quick evaluation on a sample of dev queries for faster feedback during training.
    Uses pytrec_eval for consistency with main evaluation.
    """
    model.eval()

    print(f"Running quick evaluation on {sample_size} queries...")

    # Sample a subset of queries
    all_qids = list(dev_query_to_candidates.keys())
    sample_qids = all_qids[:min(sample_size, len(all_qids))]

    # Create mini run dictionary
    run = {}

    with torch.no_grad():
        for qid in tqdm(sample_qids, desc="Quick Eval Sample"):
            candidate_pids = dev_query_to_candidates[qid]
            if not candidate_pids:
                continue

            try:
                q_embed_idx = qid_to_idx[qid]
                q_embed_np = query_embeddings[q_embed_idx]
                q_embed = torch.tensor(q_embed_np, dtype=torch.float).unsqueeze(0).to(config.DEVICE)
            except KeyError:
                continue

            candidate_p_embeds_list = []
            valid_candidate_pids = []

            # Handle different formats of candidate_pids (same as in evaluate_model_on_dev)
            for item in candidate_pids:
                pid = item[0] if isinstance(item, tuple) else item
                try:
                    p_embed_idx = pid_to_idx[pid]
                    candidate_p_embeds_list.append(passage_embeddings[p_embed_idx])
                    valid_candidate_pids.append(pid)
                except KeyError:
                    continue

            if not candidate_p_embeds_list:
                continue

            candidate_p_embeds = torch.tensor(np.array(candidate_p_embeds_list), dtype=torch.float).to(config.DEVICE)
            q_embed_expanded = q_embed.expand(candidate_p_embeds.size(0), -1)

            scores = model(q_embed_expanded, candidate_p_embeds)

            # Store scores for pytrec_eval
            run[qid] = {}
            for i, pid in enumerate(valid_candidate_pids):
                run[qid][pid] = scores[i].item()

    # Evaluate using pytrec_eval (only for available queries)
    if run:
        # Load qrels
        qrels = load_qrels(use_ir_datasets=use_ir_datasets)
        filtered_qrels = {qid: qrels[qid] for qid in run if qid in qrels}

        if filtered_qrels:
            evaluator = pytrec_eval.RelevanceEvaluator(
                filtered_qrels,
                {'mrr_cut.10', 'ndcg_cut.10', 'recall.100'}
            )

            results = evaluator.evaluate(run)

            # Calculate averages
            mrr_10 = np.mean([results[qid]['mrr_cut.10'] for qid in results])
            ndcg_10 = np.mean([results[qid]['ndcg_cut.10'] for qid in results])
            recall_100 = np.mean([results[qid]['recall.100'] for qid in results])

            print(f"Quick eval results (sample of {len(results)} queries):")
            print(f"  MRR@10: {mrr_10:.4f}")
            print(f"  nDCG@10: {ndcg_10:.4f}")
            print(f"  Recall@100: {recall_100:.4f}")

            return mrr_10
        else:
            print("No valid queries for quick evaluation")
            return 0.0
    else:
        print("No valid queries for quick evaluation")
        return 0.0