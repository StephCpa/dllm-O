from __future__ import annotations

import pytest

from dllm_order_transmission.analysis import paired_bootstrap_ci


def test_paired_bootstrap_preserves_exact_constant_difference() -> None:
    low, high = paired_bootstrap_ci(
        [1.0, 0.5, 0.25],
        [0.5, 0.0, -0.25],
        replicates=100,
        seed=7,
    )
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_paired_bootstrap_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [], replicates=10)
