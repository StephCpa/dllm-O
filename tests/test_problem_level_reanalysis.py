from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/problem_level_reanalysis.py"
SPEC = importlib.util.spec_from_file_location("problem_level_reanalysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

POLICIES = ("ar_b1", "ao_b32", "random_b32", "ef_b32")


def cell(successes_first: int, successes_second: int, answer: str = "7") -> list[dict]:
    """64 rollouts: the first 32 hold `successes_first` correct, the rest `successes_second`."""
    records = []
    for half, count in ((0, successes_first), (1, successes_second)):
        for index in range(32):
            correct = index < count
            records.append({"correct": correct, "parsed_answer": answer if correct else f"w{half}{index % 3}"})
    return records


def make_records(pattern: dict[str, tuple[int, int]]) -> tuple[dict, dict]:
    queries = {"gsm8k": [0, 1, 2], "math500": [0, 1, 2]}
    records = {}
    for task in queries:
        for row in queries[task]:
            for policy in POLICIES:
                first, second = pattern.get(policy, (16, 16))
                records[(task, row, policy)] = cell((first + row) % 33, (second + row) % 33)
    return records, queries


def test_win_loss_net_matches_pass_at_64_difference() -> None:
    records, queries = make_records({"random_b32": (0, 2), "ao_b32": (0, 0)})
    result = MODULE.coverage_win_loss(records, queries, "random_b32", "ao_b32")
    expected = statistics.mean(
        statistics.mean(
            MODULE.pass_at_k(64, MODULE.successes(records, t, r, "random_b32"), 64)
            - MODULE.pass_at_k(64, MODULE.successes(records, t, r, "ao_b32"), 64)
            for r in queries[t]
        )
        for t in queries
    )
    assert result["pooled_net_pass_at_64"] == pytest.approx(expected)


def test_joint_counts_cover_every_query() -> None:
    records, queries = make_records({})
    table = MODULE.joint_counts(records, queries, "random_b32", "ao_b32")["table"]
    for task in queries:
        assert sum(sum(row.values()) for row in table[task].values()) == len(queries[task])


def test_subset_vote_at_one_equals_pass_at_one() -> None:
    records = cell(10, 20)
    rng = np.random.default_rng(0)
    assert MODULE.subset_vote(records, 1, rng) == pytest.approx(30 / 64)


def test_equivalence_grouping_merges_equal_answers() -> None:
    records = [
        {"correct": True, "parsed_answer": "0.5"},
        {"correct": True, "parsed_answer": "1/2"},
        {"correct": False, "parsed_answer": "3"},
        {"correct": False, "parsed_answer": "3"},
        {"correct": True, "parsed_answer": "0.50"},
    ]
    equal = lambda a, b: {a, b} <= {"0.5", "1/2", "0.50"}  # noqa: E731
    assert MODULE.vote(records) == pytest.approx(0.0)
    assert MODULE.vote(records, equal) == pytest.approx(1.0)


def test_split_sample_estimates_on_held_out_rollouts_only() -> None:
    # The first half favours RND, the second half favours CF: the stratum
    # selected on the first half must show the second half's sign.
    records, queries = make_records({"random_b32": (30, 2), "ao_b32": (2, 30)})
    result = MODULE.split_sample_heterogeneity(records, queries, "random_b32", "ao_b32")
    ahead = result["strata"]["left_ahead"]
    assert sum(ahead["n_queries"].values()) == 6
    assert ahead["pass_at_1"]["point"] < 0


def test_analyze_runs_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "SUBSET_DRAWS", 5)
    monkeypatch.setattr(MODULE, "BOOTSTRAP_REPLICATES", 200)
    records, queries = make_records({"random_b32": (12, 14), "ao_b32": (20, 18)})
    report = MODULE.analyze(records, queries)
    assert set(report["pairs"]) == {"random_b32_vs_ao_b32", "ar_b1_vs_ao_b32", "ar_b1_vs_random_b32"}
    assert report["majority_vote_equivalence_grouping"].startswith("skipped")
