#!/usr/bin/env python3
"""E3: problem-level reanalysis of the held-out rollouts.

Post-outcome, exploratory, CPU-only. Nothing here replaces a frozen endpoint.
The analyses ask which queries drive the aggregate CF--RND crossing and
whether the majority-vote results are an artifact of how votes are formed:

1. the joint distribution of per-query success counts for two policies,
   binned so that probability mass moving between bins is visible;
2. the k = n = 64 coverage difference split into queries won, lost, shared,
   and unsolved; with n = k, Pass@64 is the indicator that any rollout is
   correct, so the net difference is (wins - losses) / queries;
3. split-sample heterogeneity: queries are stratified by the sign of the
   policy difference on rollouts 0-31 and the contrast is estimated on
   rollouts 32-63 only, so stratifying and estimating never use the same
   outcomes;
4. majority voting averaged over random k-subsets instead of the first k
   rollout indices, so MV@1 equals the subset-averaged Pass@1;
5. optionally, majority voting with answers grouped by mathematical
   equivalence (the pinned JustGRPO ``math_equal``) instead of exact strings,
   without reading the reference answers.

Run on the server where the raw records live:

    python scripts/problem_level_reanalysis.py --root dllm_order_transmission \\
        --output-dir reports/problem_level_reanalysis_<date> [--justgrpo-root PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from postreview_cpu_audits import TASKS, load_all, load_queries, pass_at_k  # noqa: E402

BOOTSTRAP_SEED = 20261003
BOOTSTRAP_REPLICATES = 10_000
SUBSET_DRAWS = 200
N_ROLLOUTS = 64
KS = (1, 2, 4, 8, 16, 32, 64)
COUNT_BINS = ((0, 0), (1, 1), (2, 4), (5, 16), (17, 48), (49, 63), (64, 64))
PAIRS = (("random_b32", "ao_b32"), ("ar_b1", "ao_b32"), ("ar_b1", "random_b32"))
VOTE_POLICIES = ("ar_b1", "ao_b32", "random_b32", "ef_b32")

Records = dict[tuple[str, int, str], list[dict]]
Equal = Callable[[str, str], bool]


def bootstrap(values: dict[str, list[float]], label: str) -> dict[str, float]:
    """Equal-task mean with a task-stratified percentile query bootstrap."""
    point = statistics.mean(statistics.mean(values[task]) for task in TASKS if values[task])
    seed = int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    draws = []
    for task in TASKS:
        source = np.asarray(values[task], dtype=float)
        if len(source) == 0:
            continue
        indices = rng.integers(0, len(source), size=(BOOTSTRAP_REPLICATES, len(source)))
        draws.append(source[indices].mean(axis=1))
    pooled = np.mean(np.stack(draws), axis=0)
    return {
        "point": point,
        "ci_low": float(np.quantile(pooled, 0.025)),
        "ci_high": float(np.quantile(pooled, 0.975)),
        "n_queries": {task: len(values[task]) for task in TASKS},
    }


def successes(records: Records, task: str, row: int, policy: str, indices=None) -> int:
    cell = records[(task, row, policy)]
    chosen = cell if indices is None else [cell[i] for i in indices]
    return sum(bool(record["correct"]) for record in chosen)


def count_bin(count: int) -> str:
    for low, high in COUNT_BINS:
        if low <= count <= high:
            return f"{low}" if low == high else f"{low}-{high}"
    raise ValueError(count)


def joint_counts(records: Records, queries, left: str, right: str) -> dict:
    labels = [count_bin(low) for low, _ in COUNT_BINS]
    table = {task: {a: {b: 0 for b in labels} for a in labels} for task in TASKS}
    for task in TASKS:
        for row in queries[task]:
            a = count_bin(successes(records, task, row, left))
            b = count_bin(successes(records, task, row, right))
            table[task][a][b] += 1
    return {"rows": left, "columns": right, "bins": labels, "table": table}


def coverage_win_loss(records: Records, queries, left: str, right: str) -> dict:
    """Decompose the Pass@64 difference left - right into won and lost queries."""
    result = {}
    for task in TASKS:
        counts = {"wins": 0, "losses": 0, "both": 0, "neither": 0}
        for row in queries[task]:
            a = successes(records, task, row, left) > 0
            b = successes(records, task, row, right) > 0
            key = "both" if a and b else "wins" if a else "losses" if b else "neither"
            counts[key] += 1
        counts["net_pass_at_64"] = (counts["wins"] - counts["losses"]) / len(queries[task])
        result[task] = counts
    result["pooled_net_pass_at_64"] = statistics.mean(result[task]["net_pass_at_64"] for task in TASKS)
    return result


def split_sample_heterogeneity(records: Records, queries, left: str, right: str) -> dict:
    """Stratify on rollouts 0-31, estimate on rollouts 32-63."""
    first, second = range(0, 32), range(32, 64)
    strata = {"left_ahead": {t: [] for t in TASKS}, "tied": {t: [] for t in TASKS},
              "right_ahead": {t: [] for t in TASKS}}
    first_diff, second_diff = [], []
    for task in TASKS:
        for row in queries[task]:
            d1 = successes(records, task, row, left, first) - successes(records, task, row, right, first)
            d2 = successes(records, task, row, left, second) - successes(records, task, row, right, second)
            first_diff.append(d1)
            second_diff.append(d2)
            name = "left_ahead" if d1 > 0 else "right_ahead" if d1 < 0 else "tied"
            strata[name][task].append(row)
    result = {"stratified_on": "rollouts 0-31", "estimated_on": "rollouts 32-63", "strata": {}}
    for name, rows in strata.items():
        result["strata"][name] = {"n_queries": {task: len(rows[task]) for task in TASKS}}
        if not any(rows.values()):
            continue
        for k in (1, 16):
            values = {
                task: [
                    pass_at_k(32, successes(records, task, row, left, second), k)
                    - pass_at_k(32, successes(records, task, row, right, second), k)
                    for row in rows[task]
                ]
                for task in TASKS
            }
            if all(values[task] for task in TASKS):
                result["strata"][name][f"pass_at_{k}"] = bootstrap(values, f"split|{left}|{right}|{name}|{k}")
    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    result["half_to_half_rank_correlation"] = float(np.corrcoef(ranks(first_diff), ranks(second_diff))[0, 1])
    return result


def vote(records: list[dict], equal: Equal | None = None) -> float:
    """Tie-aware majority vote; answers grouped by string or by an equivalence."""
    groups: list[tuple[str, list[bool]]] = []
    for record in records:
        answer = record.get("parsed_answer")
        if answer is None or str(answer).strip() == "":
            continue
        answer = str(answer)
        for representative, flags in groups:
            if representative == answer or (equal is not None and equal(representative, answer)):
                flags.append(bool(record["correct"]))
                break
        else:
            groups.append((answer, [bool(record["correct"])]))
    if not groups:
        return 0.0
    largest = max(len(flags) for _, flags in groups)
    return statistics.mean(statistics.mean(flags) for _, flags in groups if len(flags) == largest)


def subset_vote(cell: list[dict], k: int, rng: np.random.Generator, equal: Equal | None = None) -> float:
    """Expected majority-vote accuracy over k-subsets of the stored rollouts."""
    if k == 1:
        return statistics.mean(vote([record], equal) for record in cell)
    if k == len(cell):
        return vote(cell, equal)
    return statistics.mean(
        vote([cell[i] for i in rng.choice(len(cell), size=k, replace=False)], equal)
        for _ in range(SUBSET_DRAWS)
    )


def majority_vote_analysis(records: Records, queries, equal: Equal | None = None, tag: str = "string") -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_query = {
        policy: {k: {task: [] for task in TASKS} for k in KS} for policy in VOTE_POLICIES
    }
    for task in TASKS:
        for row in queries[task]:
            for policy in VOTE_POLICIES:
                cell = records[(task, row, policy)]
                for k in KS:
                    per_query[policy][k][task].append(subset_vote(cell, k, rng, equal))
    pooled = {
        policy: {str(k): statistics.mean(statistics.mean(per_query[policy][k][t]) for t in TASKS) for k in KS}
        for policy in VOTE_POLICIES
    }
    contrasts = {}
    for left, right in (("ar_b1", "random_b32"), ("ao_b32", "random_b32"), ("ar_b1", "ao_b32")):
        contrasts[f"{left}_minus_{right}"] = {
            str(k): bootstrap(
                {t: [a - b for a, b in zip(per_query[left][k][t], per_query[right][k][t])] for t in TASKS},
                f"vote|{tag}|{left}|{right}|{k}",
            )
            for k in KS
        }
    return {"grouping": tag, "subset_draws": SUBSET_DRAWS, "pooled": pooled, "paired_contrasts": contrasts}


def load_equivalence(justgrpo_root: Path) -> Equal:
    sys.path.insert(0, str(justgrpo_root))
    from utils.grader import math_equal  # type: ignore[import-not-found]

    cache: dict[tuple[str, str], bool] = {}

    def equal(a: str, b: str) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key not in cache:
            cache[key] = bool(math_equal(a, b, timeout=True))
        return cache[key]

    return equal


def analyze(records: Records, queries, equal: Equal | None = None) -> dict:
    report = {
        "analysis_status": "post_outcome_exploratory_cpu_only",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "pairs": {},
        "majority_vote_string_grouping": majority_vote_analysis(records, queries),
    }
    for left, right in PAIRS:
        report["pairs"][f"{left}_vs_{right}"] = {
            "joint_success_counts": joint_counts(records, queries, left, right),
            "coverage_win_loss_k64": coverage_win_loss(records, queries, left, right),
            "split_sample_heterogeneity": split_sample_heterogeneity(records, queries, left, right),
        }
    if equal is not None:
        report["majority_vote_equivalence_grouping"] = majority_vote_analysis(
            records, queries, equal, tag="math_equal"
        )
    else:
        report["majority_vote_equivalence_grouping"] = "skipped: no --justgrpo-root given"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--justgrpo-root", type=Path, default=None)
    args = parser.parse_args()
    queries = load_queries(args.root)
    records = load_all(args.root, queries)
    equal = load_equivalence(args.justgrpo_root) if args.justgrpo_root else None
    report = analyze(records, queries, equal)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "problem_level_reanalysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
