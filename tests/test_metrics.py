import pytest

from dllm_order_transmission.metrics import aggregate_pass_at_k, estimate_pass_at_k


@pytest.mark.parametrize("n,k", [(32, 1), (32, 16), (32, 32)])
def test_pass_at_k_edge_cases(n: int, k: int) -> None:
    assert estimate_pass_at_k(n, 0, k) == 0.0
    assert estimate_pass_at_k(n, n, k) == 1.0


def test_pass_at_one_matches_success_rate() -> None:
    assert estimate_pass_at_k(32, 8, 1) == pytest.approx(0.25)


def test_pass_at_n_is_any_success() -> None:
    assert estimate_pass_at_k(32, 1, 32) == 1.0
    assert estimate_pass_at_k(32, 0, 32) == 0.0


def test_aggregate_is_query_average() -> None:
    assert aggregate_pass_at_k([0, 32], n=32, k=1) == pytest.approx(0.5)
