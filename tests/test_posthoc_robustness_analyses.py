import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "posthoc_robustness_analyses.py"
)
SPEC = importlib.util.spec_from_file_location("posthoc_robustness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_log2_slope_decomposition_identity():
    y1, y8, y16, y32 = 0.79746, 0.75960, 0.75084, 0.75325
    direct = (-3 * y1 + y16 + 2 * y32) / 14
    parallel_mean = (y8 + y16 + y32) / 3
    decomposed = 3 * (parallel_mean - y1) / 14 + (y32 - y8) / 14
    assert abs(direct - decomposed) < 1e-15


def test_pass_at_k_edges():
    assert MODULE.pass_at_k(64, 0, 16) == 0.0
    assert MODULE.pass_at_k(64, 64, 16) == 1.0
    assert MODULE.pass_at_k(64, 16, 1) == 0.25


def test_tie_aware_spearman():
    assert MODULE.spearman([0, 0, 1, 1], [1, 1, 2, 2]) == 1.0
