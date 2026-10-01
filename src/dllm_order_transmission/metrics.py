"""Query-level metrics for repeated reasoning rollouts."""

from __future__ import annotations

import math
from collections.abc import Iterable


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Return the standard unbiased Pass@k estimate for one query."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def aggregate_pass_at_k(success_counts: Iterable[int], n: int, k: int) -> float:
    """Average query-level Pass@k estimates without treating rollouts as units."""
    counts = list(success_counts)
    if not counts:
        raise ValueError("success_counts must not be empty")
    return sum(estimate_pass_at_k(n, c, k) for c in counts) / len(counts)
