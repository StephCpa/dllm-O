from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/postreview_cpu_audits.py"
SPEC = importlib.util.spec_from_file_location("postreview_cpu_audits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pass_at_k_boundaries() -> None:
    assert MODULE.pass_at_k(64, 0, 32) == 0.0
    assert MODULE.pass_at_k(64, 64, 32) == 1.0


def test_majority_vote_uses_fractional_tie_breaking() -> None:
    records = [
        {"parsed_answer": "a", "correct": True},
        {"parsed_answer": "a", "correct": True},
        {"parsed_answer": "b", "correct": False},
        {"parsed_answer": "b", "correct": False},
    ]
    assert MODULE.majority_vote_correct(records) == pytest.approx(0.5)
    assert MODULE.majority_vote_correct([]) == 0.0


def test_generation_accounting_closes() -> None:
    accounting = MODULE.generation_accounting()
    assert accounting["total"] == 86_400


def test_math_difficulty_checks_group_frozen_strata() -> None:
    metadata = {
        "math500": {
            3: {"stratum": "Algebra|Level 1"},
            7: {"stratum": "Geometry|Level 2"},
        }
    }
    records = {}
    for row, successes in ((3, 64), (7, 0)):
        for policy in MODULE.POLICIES:
            records[("math500", row, policy)] = [
                {"correct": rollout < successes} for rollout in range(64)
            ]

    result = MODULE.math_difficulty_checks(records, metadata)

    assert result["levels"]["Level 1"]["queries"] == 1
    assert result["levels"]["Level 1"]["policies"]["ar_b1"]["pass_at_1"] == 1.0
    assert result["levels"]["Level 2"]["policies"]["ar_b1"]["pass_at_1"] == 0.0
