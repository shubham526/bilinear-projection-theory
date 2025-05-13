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
    Generates the indicator vector e_I0 in {-1, +1}^n_dim as per Definition 3.3.
    (e_I0)_k = +1 if k in I0_indices, -1 if k not in I0_indices.
    """
    e_I0 = -torch.ones(n_dim, dtype=torch.float32)
    for idx in I0_indices:
        e_I0[idx] = 1.0
    return e_I0


def generate_structured_agreement_docs(query_q, e_I0_hypercube):
    """
    Generates D(q, I0) = {q, q⊙e_I0, -q, -(q⊙e_I0)} per Definition 3.4
    where ⊙ is element-wise product (Hadamard product).

    The task requires ranking documents that agree with q on I0 above those that disagree.

    Args:
        query_q: Query vector in {-1, +1}^n
        e_I0_hypercube: Indicator vector for index set I0

    Returns:
        docs_agree: Documents that agree with q on I0 dimensions [q, q⊙e_I0]
        docs_disagree: Documents that disagree with q on I0 dimensions [-q, -(q⊙e_I0)]
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


def generate_challenging_patterns(n_dim):
    """
    Generate challenging query-I0 patterns that are particularly difficult for WDP models.
    These patterns highlight when different I0 sets require conflicting weight priorities.
    """
    patterns = []

    if n_dim >= 3:
        # Pattern 1: Adjacent features
        patterns.append({
            "q": torch.tensor([1, 1, 1] + [-1] * (n_dim - 3), dtype=torch.float32),
            "I0": (0, 1),
            "name": "Adjacent features (0,1)",
            "description": "Requires weighting first two dimensions highly"
        })

        # Pattern 2: Skip one feature
        patterns.append({
            "q": torch.tensor([1, -1, 1] + [-1] * (n_dim - 3), dtype=torch.float32),
            "I0": (0, 2),
            "name": "Skip one feature (0,2)",
            "description": "Requires weighting dimensions 0 and 2, but not 1"
        })

        # Pattern 3: End features
        if n_dim >= 4:
            patterns.append({
                "q": torch.tensor([1, -1, -1, 1] + [-1] * (n_dim - 4), dtype=torch.float32),
                "I0": (0, 3),
                "name": "End features (0,3)",
                "description": "Requires weighting first and last relevant dimensions"
            })

    # Pattern 4: First two (fallback for n_dim < 3)
    if n_dim >= 2:
        patterns.append({
            "q": torch.tensor([1, 1] + [-1] * (n_dim - 2), dtype=torch.float32),
            "I0": (0, 1),
            "name": "Standard first two",
            "description": "Standard test case for first two dimensions"
        })

    return patterns


def compute_query_document_agreement(query, doc, I0_indices):
    """
    Compute how many dimensions in I0 the query and document agree on.
    Agreement means both have the same sign (both +1 or both -1).
    """
    agreements = 0
    for idx in I0_indices:
        if query[idx] * doc[idx] > 0:  # Same sign
            agreements += 1
    return agreements


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
    for i, d in enumerate(agree_docs):
        print(f"  d{i + 1}: {d}")
    print("\nDisagree Set:")
    for i, d in enumerate(disagree_docs):
        print(f"  d{i + 1}: {d}")

    print(f"\nAll possible I0 for n=3 (k=2): {generate_all_possible_I0_indices(3, 2)}")

    # Test challenging patterns
    print("\nChallenging patterns:")
    patterns = generate_challenging_patterns(n)
    for pattern in patterns:
        print(f"  {pattern['name']}: {pattern['description']}")

    # Test agreement computation
    print(f"\nAgreement between q_test and first agree doc on I0 {I0_test_indices}:")
    agreement = compute_query_document_agreement(q_test, agree_docs[0], I0_test_indices)
    print(f"  Agreement count: {agreement} (should be {len(I0_test_indices)})")