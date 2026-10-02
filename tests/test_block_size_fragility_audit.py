from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/block_size_fragility_audit.py"
SPEC = importlib.util.spec_from_file_location("block_size_fragility_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_values() -> dict[str, list[float]]:
    # Three strongly negative GSM8K queries, one on MATH-500, and small positive noise.
    gsm8k = [-0.8, -0.6, -0.4] + [0.001] * 97
    math500 = [-0.5] + [0.0] * 99
    return {"gsm8k": gsm8k, "math500": math500}


def test_pooled_point_is_equal_task_mean() -> None:
    values = {task: np.asarray(v) for task, v in synthetic_values().items()}
    expected = (np.mean(values["gsm8k"]) + np.mean(values["math500"])) / 2
    assert MODULE.pooled_point(values) == pytest.approx(expected)


def test_bootstrap_is_deterministic_and_brackets_point() -> None:
    values = {task: np.asarray(v) for task, v in synthetic_values().items()}
    first = MODULE.pooled_bootstrap(values, seed=7)
    second = MODULE.pooled_bootstrap(values, seed=7)
    assert first == second
    assert first["ci_low"] <= first["point"] <= first["ci_high"]


def test_removal_order_ranks_across_tasks() -> None:
    values = {task: np.asarray(v) for task, v in synthetic_values().items()}
    order = MODULE.removal_order(values, most_negative_first=True)
    assert [(task, index) for task, index, _ in order[:4]] == [
        ("gsm8k", 0),
        ("gsm8k", 1),
        ("math500", 0),
        ("gsm8k", 2),
    ]


def test_audit_counts_removals_and_concentration() -> None:
    report = MODULE.audit(synthetic_values(), max_removed=6)
    curve = report["most_negative_removed_first"]["curve"]
    assert curve[0]["removed"] == 0 and curve[0]["last_removed_effect"] is None
    assert curve[4]["removed_gsm8k"] == 3 and curve[4]["removed_math500"] == 1
    # After the four negative queries are gone only nonnegative effects remain.
    assert curve[4]["point"] > 0
    assert report["most_negative_removed_first"]["first_removed_where_point_is_nonnegative"] <= 4
    shares = report["concentration"]["share_by_most_negative"]
    assert shares["1"] == pytest.approx(-0.8 / (-2.3 + 0.097))
