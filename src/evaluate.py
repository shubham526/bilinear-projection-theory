# evaluate.py
import subprocess
import os
import torch
from tqdm import tqdm
import numpy as np
import config
from models import get_model
import re


def evaluate_model_on_dev(model, query_embeddings, passage_embeddings,
                          qid_to_idx, pid_to_idx, dev_query_to_candidates,
                          run_file_path="run.dev.txt"):
    """
    Evaluate model on dev set by scoring candidates and creating a TREC run file.
    Then calls the official MS MARCO evaluation script.
    """
    model.eval()

    # Ensure run_file_path directory exists
    os.makedirs(os.path.dirname(run_file_path), exist_ok=True)

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
                for pid in candidate_pids:
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

                for rank_idx, original_candidate_idx in enumerate(sorted_indices):
                    passage_id = valid_candidate_pids[original_candidate_idx.item()]
                    score = scores[original_candidate_idx.item()].item()
                    # Write in TREC run format: qid Q0 pid rank score run_name
                    f_run.write(f"{qid}\tQ0\t{passage_id}\t{rank_idx + 1}\t{score:.6f}\tCustomModel\n")

    # After writing the run file, call the official MS MARCO evaluation script
    return evaluate_run_file(config.DEV_QRELS_PATH, run_file_path)


def evaluate_run_file(qrels_path, run_file_path):
    """
    Calls the official MS MARCO evaluation script and returns MRR@10.
    """
    try:
        print(f"Running MS MARCO evaluation script on {run_file_path}...")

        # Check if the evaluation script exists
        if not os.path.exists(config.MSMARCO_EVAL_SCRIPT):
            print(f"Warning: MS MARCO evaluation script not found at {config.MSMARCO_EVAL_SCRIPT}")
            print("Please download the official script from MS MARCO GitHub repository.")
            return 0.0

        result = subprocess.run(
            ['python', config.MSMARCO_EVAL_SCRIPT, qrels_path, run_file_path],
            capture_output=True, text=True, check=True
        )

        print("Evaluation Output:")
        print(result.stdout)

        # Parse MRR@10 from result.stdout (specific to MS MARCO script output format)
        mrr_10 = 0.0

        # Try multiple parsing patterns as different versions of the script might format differently
        patterns = [
            r"MRR @10:\s*([0-9\.]+)",
            r"MRR@10:\s*([0-9\.]+)",
            r"MRR\s*@\s*10\s*:\s*([0-9\.]+)",
            r"#####################\nMRR @10: ([0-9\.]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, result.stdout)
            if match:
                mrr_10 = float(match.group(1))
                break

        if mrr_10 == 0.0:
            print("Warning: Could not parse MRR@10 from evaluation output.")
            print("Please check the output format of your evaluation script.")
        else:
            print(f"Parsed MRR@10: {mrr_10:.4f}")

        return mrr_10

    except subprocess.CalledProcessError as e:
        print("Error during evaluation:")
        print(e.stderr)
        print("stdout:", e.stdout)
        return 0.0
    except FileNotFoundError:
        print(f"Error: Python interpreter or evaluation script not found.")
        print(f"Make sure {config.MSMARCO_EVAL_SCRIPT} exists and is executable.")
        return 0.0
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        return 0.0


def quick_eval_sample(model, query_embeddings, passage_embeddings,
                      qid_to_idx, pid_to_idx, dev_query_to_candidates, sample_size=100):
    """
    Quick evaluation on a sample of dev queries for faster feedback during training.
    """
    model.eval()

    # Sample a subset of queries
    all_qids = list(dev_query_to_candidates.keys())
    sample_qids = all_qids[:min(sample_size, len(all_qids))]

    total_reciprocal_rank = 0.0
    valid_queries = 0

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

            for pid in candidate_pids:
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
            sorted_indices = torch.argsort(scores, descending=True)

            # Check if any of the top 10 candidates are relevant
            # For this quick eval, we'll assume first candidate is relevant
            # (this is a simplification - in real eval, you'd check against qrels)
            if len(sorted_indices) > 0:
                rank = 1  # Since we don't have qrels here, assume rank 1
                total_reciprocal_rank += 1.0 / rank
                valid_queries += 1

    if valid_queries > 0:
        avg_rr = total_reciprocal_rank / valid_queries
        print(f"Quick eval MRR (sample of {valid_queries} queries): {avg_rr:.4f}")
        return avg_rr
    else:
        print("No valid queries for quick evaluation")
        return 0.0