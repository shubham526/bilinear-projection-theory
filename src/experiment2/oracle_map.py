import collections
import argparse


def parse_qrels(qrels_file_path):
    """
    Parses a qrels file.
    Assumes format: query_id <ignored> doc_id relevance_judgment
    Relevance is > 0 for relevant documents.

    Returns:
        dict: query_id -> set of relevant doc_ids
    """
    qrels = collections.defaultdict(set)
    try:
        with open(qrels_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 4:
                    print(f"Warning: Skipping malformed line {line_num} in qrels file: {line.strip()}")
                    continue
                query_id = parts[0]
                doc_id = parts[2]
                try:
                    relevance = int(parts[3])
                    if relevance > 0:
                        qrels[query_id].add(doc_id)
                except ValueError:
                    print(
                        f"Warning: Skipping line {line_num} in qrels file due to non-integer relevance: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Qrels file not found at {qrels_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing qrels file '{qrels_file_path}': {e}")
        return None
    return qrels


def parse_run_file(run_file_path):
    """
    Parses a TREC run file.
    Assumes format: query_id <Q0_literal> doc_id <rank> <score> <run_name>
    We only need query_id and the set of doc_ids retrieved for that query.

    Returns:
        dict: query_id -> set of retrieved doc_ids
    """
    run_retrieved_docs = collections.defaultdict(set)
    try:
        with open(run_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 3:  # Need at least query_id, Q0, doc_id
                    print(f"Warning: Skipping malformed line {line_num} in run file: {line.strip()}")
                    continue
                query_id = parts[0]
                doc_id = parts[2]
                run_retrieved_docs[query_id].add(doc_id)
    except FileNotFoundError:
        print(f"Error: Run file not found at {run_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing run file '{run_file_path}': {e}")
        return None
    return run_retrieved_docs


def calculate_oracle_map(run_file_path, qrels_file_path):
    """
    Calculates the oracle MAP for a given TREC run file and qrels file.
    This represents the maximum possible MAP achievable by reranking the documents
    present in the run file.
    """
    qrels_data = parse_qrels(qrels_file_path)
    if qrels_data is None:
        print("Exiting due to qrels parsing error.")
        return 0.0

    run_data_retrieved_docs = parse_run_file(run_file_path)
    if run_data_retrieved_docs is None:
        print("Exiting due to run file parsing error.")
        return 0.0

    average_precisions = []

    if not run_data_retrieved_docs:
        print("Warning: Run file seems empty or no queries were parsed from it.")
        return 0.0

    processed_queries_count = 0
    # Iterate over queries present in the run file
    for query_id in run_data_retrieved_docs:
        retrieved_docs_for_query = run_data_retrieved_docs[query_id]

        # Relevant documents for this query according to qrels
        relevant_docs_in_qrels_for_query = qrels_data.get(query_id, set())

        # Total number of relevant documents for this query in the qrels (R for AP calculation)
        R_total_relevant_in_qrels = len(relevant_docs_in_qrels_for_query)

        if R_total_relevant_in_qrels == 0:
            # If there are no relevant documents for this query in qrels, AP is 0.
            # This query still counts towards the MAP calculation (as a query with 0 AP).
            average_precisions.append(0.0)
            print(f"Query {query_id}: No relevant documents in qrels. AP = 0.0")
            processed_queries_count += 1
            continue

        # Identify which of the *retrieved* documents are actually relevant
        num_retrieved_and_relevant = 0
        for doc_id in retrieved_docs_for_query:
            if doc_id in relevant_docs_in_qrels_for_query:
                num_retrieved_and_relevant += 1

        if num_retrieved_and_relevant == 0:
            average_precisions.append(0.0)
            print(
                f"Query {query_id}: No relevant documents retrieved out of {R_total_relevant_in_qrels} total relevant in qrels. AP = 0.0")
            processed_queries_count += 1
            continue

        # Calculate AP for this oracle ranking:
        # The numerator of AP is sum(P@k * rel_k).
        # In the oracle ranking of *retrieved* documents, the first 'num_retrieved_and_relevant'
        # documents are relevant.
        # P@k for the k-th relevant document (at rank k in this ideal list) is k/k = 1.
        # So, the sum of (P@k * rel_k) becomes the sum of (1) for k from 1 to num_retrieved_and_relevant.
        # This sum is simply num_retrieved_and_relevant.
        sum_of_precisions_times_rel = float(num_retrieved_and_relevant)

        ap_for_query = sum_of_precisions_times_rel / R_total_relevant_in_qrels
        average_precisions.append(ap_for_query)
        print(f"Query {query_id}: Retrieved {len(retrieved_docs_for_query)} docs. "
              f"{num_retrieved_and_relevant} are relevant (out of {R_total_relevant_in_qrels} total relevant in qrels). "
              f"Oracle AP numerator = {sum_of_precisions_times_rel:.4f}. Oracle AP = {ap_for_query:.4f}")
        processed_queries_count += 1

    if not processed_queries_count:
        print(
            "No queries from the run file were processed (e.g., run file empty or queries not in qrels). MAP cannot be calculated.")
        return 0.0
    if not average_precisions:  # Should be covered by processed_queries_count check
        print("No APs could be calculated. MAP is 0.0")
        return 0.0

    # MAP is the mean of APs for all queries in the run file.
    # If a query from the run file is not in qrels, its R_total_relevant_in_qrels is 0, leading to AP=0.
    # The number of queries for MAP calculation is the number of unique queries in the run file.
    oracle_map = sum(average_precisions) / len(run_data_retrieved_docs) if run_data_retrieved_docs else 0.0
    return oracle_map


if __name__ == '__main__':
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate Oracle MAP for a TREC run file.")
    parser.add_argument("--qrels", help="Path to the qrels (relevance judgments) file.", required=True)
    parser.add_argument("--run", help="Path to the TREC run file.", required=True)

    args = parser.parse_args()

    qrels_file_path = args.qrels
    run_file_path = args.run

    print(f"Calculating Oracle MAP for run file '{run_file_path}' and qrels file '{qrels_file_path}'\n")

    oracle_map_value = calculate_oracle_map(run_file_path, qrels_file_path)

    print(f"\n--------------------------------------------------")
    print(f"Oracle MAP (Upper Bound for Reranking of this run): {oracle_map_value:.4f}")
    print(f"Calculated over {len(parse_run_file(run_file_path) or {})} queries found in the run file.")
    print(f"--------------------------------------------------")
