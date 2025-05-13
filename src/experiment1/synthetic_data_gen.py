import torch
import numpy as np
import random
import itertools


def generate_random_query_vector(n_dim):
    """Generates a random query vector q in {-1, +1}^n_dim."""
    # np.random.choice is efficient for this
    return torch.tensor(np.random.choice([-1, 1], size=n_dim), dtype=torch.float32)


def generate_random_I0_indices(n_dim, num_indices=2):
    """Generates a random set of 'num_indices' distinct indices from [0, n_dim-1]."""
    if n_dim < num_indices:
        raise ValueError(f"n_dim ({n_dim}) must be >= num_indices ({num_indices})")
    return tuple(sorted(random.sample(range(n_dim), num_indices)))


def generate_indicator_e_I_hypercube(n_dim, I0_indices):
    """
    Generates the indicator vector e_I0 in {-1, +1}^n_dim as per Def 3.3.
    (e_I0)_k = +1 if k in I0_indices, -1 if k not in I0_indices.
    """
    e_I0 = -torch.ones(n_dim, dtype=torch.float32)
    for idx in I0_indices:
        e_I0[idx] = 1.0
    return e_I0


def generate_structured_agreement_docs(query_q, e_I0_hypercube):
    """
    Generates the document multiset D(q, I0) = {q, q*e_I0, -q, -(q*e_I0)}
    where * is element-wise product (Hadamard product).
    Returns:
        docs_agree: list [d_agree1, d_agree2]
        docs_disagree: list [d_disagree1, d_disagree2]
    """
    # Element-wise product
    q_odot_e_I0 = query_q * e_I0_hypercube

    docs_agree = [query_q.clone(), q_odot_e_I0.clone()]
    docs_disagree = [-query_q.clone(), -q_odot_e_I0.clone()]

    return docs_agree, docs_disagree


def generate_all_possible_queries(n_dim):
    """Generates all 2^n_dim possible query vectors in {-1, +1}^n_dim."""
    if n_dim > 10:  # Adjust limit as needed to avoid excessive computation
        print(f"Warning: Generating all 2^{n_dim} queries can be very slow. Consider sampling.")

    # Create all combinations of -1 and 1 for n_dim positions
    # Each row in product is one query vector
    for p in itertools.product([-1, 1], repeat=n_dim):
        yield torch.tensor(p, dtype=torch.float32)


def generate_all_possible_I0_indices(n_dim, num_indices=2):
    """Generates all possible combinations of 'num_indices' from n_dim."""
    return list(itertools.combinations(range(n_dim), num_indices))


if __name__ == '__main__':
    # Example Usage
    n = 4
    q_test = generate_random_query_vector(n)
    print(f"Random query q (n={n}):\n{q_test}")

    I0_test_indices = generate_random_I0_indices(n, 2)
    print(f"Random I0 indices: {I0_test_indices}")

    e_I0_test = generate_indicator_e_I_hypercube(n, I0_test_indices)
    print(f"Indicator e_I0 for {I0_test_indices}:\n{e_I0_test}")

    agree_docs, disagree_docs = generate_structured_agreement_docs(q_test, e_I0_test)
    print("\nAgree Set:")
    for d in agree_docs:
        print(d)
    print("\nDisagree Set:")
    for d in disagree_docs:
        print(d)

    print(f"\nAll possible I0 for n=3 (k=2): {generate_all_possible_I0_indices(3, 2)}")

    # count = 0
    # for q_small in generate_all_possible_queries(3):
    #     count +=1
    # print(f"\nNumber of queries for n=3: {count}")