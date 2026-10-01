"""Post-unblinding robustness analyses for the dLLM scheduling study.

All analyses in this file are EXPLORATORY. They do not alter any frozen
endpoint or verdict. The script addresses three interpretation questions:

1. How much of the Phase 2 P2 slope is leverage from the B=1 endpoint?
2. How does the EF-vs-AO conclusion change with the sampling-budget grid?
3. Does the EF-vs-AO pass@1 gap persist in length-comparable rollouts?
4. Does matched AO-vs-random evidence hide a precision-coverage crossing?

It also reports the outcome-conditioned P1 contrast among queries with at
least one success in either endpoint arm. That subset result is descriptive
and must not replace the intention-to-treat P1 estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable


TASKS = ("gsm8k", "math500")
B_VALUES = (1, 8, 16, 32)
POLICY_BY_B = {1: "ar_b1", 8: "ao_b8", 16: "ao_b16", 32: "ao_b32"}
PRIMARY_GRID = (4, 8, 16, 32, 64)
LOW_GRID = (2, 4, 8, 16)
HIGH_GRID = (16, 32, 64)
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_REPLICATES = 10_000


def pass_at_k(n: int, successes: int, k: int) -> float:
    if not 0 <= successes <= n:
        raise ValueError("successes must be between zero and n")
    if not 1 <= k <= n:
        raise ValueError("k must be between one and n")
    if n - successes < k:
        return 1.0
    return 1.0 - math.comb(n - successes, k) / math.comb(n, k)


def coverage(successes: int, n: int, grid: tuple[int, ...]) -> float:
    return statistics.mean(pass_at_k(n, successes, k) for k in grid)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must be nonempty and equally sized")
    lm = statistics.mean(left)
    rm = statistics.mean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def stratified_summary(
    values: dict[str, list[float]],
    *,
    label: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    if any(not values[task] for task in TASKS):
        raise ValueError(f"empty task stratum for {label}")
    point = statistics.mean(statistics.mean(values[task]) for task in TASKS)
    rng = random.Random(f"{BOOTSTRAP_SEED}|{label}")
    draws = []
    for _ in range(replicates):
        task_means = []
        for task in TASKS:
            source = values[task]
            task_means.append(
                statistics.mean(source[rng.randrange(len(source))] for _ in source)
            )
        draws.append(statistics.mean(task_means))
    draws.sort()
    return {
        "point": point,
        "low": percentile(draws, 0.025),
        "high": percentile(draws, 0.975),
        "n_gsm8k": len(values["gsm8k"]),
        "n_math500": len(values["math500"]),
    }


def load_manifest(root: Path) -> tuple[dict, dict[str, list[int]]]:
    manifest = json.loads(
        (root / "protocols/phase1_split_manifest_v1.json").read_text()
    )
    queries = {
        task: sorted(
            int(entry["row_index"])
            for entry in manifest["splits"][task]["heldout"]
        )
        for task in TASKS
    }
    return manifest, queries


def phase1_record_path(
    root: Path, task: str, policy: str, row: int, rollout: int
) -> Path:
    if rollout < 16:
        run_kind = "screening"
    elif policy in ("ao_b8", "ao_b16"):
        run_kind = "phase2-extension"
    else:
        run_kind = "primary-extension"
    return (
        root
        / "outputs/dllm_order_phase1/phase1-v1/heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def condition_a_record_path(root: Path, task: str, row: int, rollout: int) -> Path:
    return (
        root
        / "outputs/dllm_order_condition_a/condition-a-v1/condition-a-formal"
        / task
        / "ef_b32"
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(json.loads(path.read_text()))
    return records


def phase2_leverage(root: Path, queries: dict[str, list[int]]) -> dict:
    per_query: dict[str, list[dict]] = {task: [] for task in TASKS}
    zero_cells = 0
    all_four_zero_queries = 0
    endpoint_inactive_queries = 0
    for task in TASKS:
        for row in queries[task]:
            successes = {}
            scores = {}
            for block_size in B_VALUES:
                policy = POLICY_BY_B[block_size]
                records = load_records(
                    [
                        phase1_record_path(root, task, policy, row, rollout)
                        for rollout in range(64)
                    ]
                )
                successes[block_size] = sum(bool(record["correct"]) for record in records)
                zero_cells += successes[block_size] == 0
                scores[block_size] = coverage(successes[block_size], 64, PRIMARY_GRID)

            y1, y8, y16, y32 = (scores[block_size] for block_size in B_VALUES)
            parallel_mean = statistics.mean((y8, y16, y32))
            p1 = y32 - y1
            p2 = (-3 * y1 + y16 + 2 * y32) / 14
            step_component = 3 * (parallel_mean - y1) / 14
            within_parallel_component = (y32 - y8) / 14
            if all(successes[block_size] == 0 for block_size in B_VALUES):
                all_four_zero_queries += 1
            endpoint_active = successes[1] > 0 or successes[32] > 0
            if not endpoint_active:
                endpoint_inactive_queries += 1
            per_query[task].append(
                {
                    "row": row,
                    "successes": successes,
                    "scores": scores,
                    "p1": p1,
                    "p2": p2,
                    "step_component": step_component,
                    "within_parallel_component": within_parallel_component,
                    "endpoint_active": endpoint_active,
                }
            )

    pooled_rows = per_query["gsm8k"] + per_query["math500"]
    p1 = [row["p1"] for row in pooled_rows]
    p2 = [row["p2"] for row in pooled_rows]
    means = {
        str(block_size): statistics.mean(
            row["scores"][block_size] for row in pooled_rows
        )
        for block_size in B_VALUES
    }
    mean_step = statistics.mean(row["step_component"] for row in pooled_rows)
    mean_within = statistics.mean(
        row["within_parallel_component"] for row in pooled_rows
    )
    mean_p2 = statistics.mean(p2)
    active_values = {
        task: [row["p1"] for row in per_query[task] if row["endpoint_active"]]
        for task in TASKS
    }
    return {
        "pooled_coverage_auc_by_block_size": means,
        "p1_p2_dependence": {
            "pearson": pearson(p1, p2),
            "spearman_tie_aware": spearman(p1, p2),
        },
        "p2_exact_decomposition": {
            "mean_p2": mean_p2,
            "b1_to_parallel_step_component": mean_step,
            "within_parallel_endpoint_component": mean_within,
            "identity_residual": mean_p2 - mean_step - mean_within,
            "absolute_fraction_from_step": (
                abs(mean_step) / (abs(mean_step) + abs(mean_within))
            ),
            "formula": (
                "beta_log2 = 3*(mean(y8,y16,y32)-y1)/14 + (y32-y8)/14"
            ),
        },
        "zero_information_audit": {
            "zero_cells_of_800": zero_cells,
            "queries_all_four_cells_zero_of_200": all_four_zero_queries,
            "queries_b1_and_b32_both_zero_of_200": endpoint_inactive_queries,
        },
        "p1_endpoint_active_subset": {
            "warning": (
                "Outcome-conditioned exploratory subset; never substitute for frozen P1."
            ),
            "summary": stratified_summary(
                active_values, label="phase2-p1-endpoint-active"
            ),
            "per_task_points": {
                task: statistics.mean(active_values[task]) for task in TASKS
            },
        },
    }


def contrast_summary(
    rows: dict[str, list[dict]],
    value: Callable[[dict], float],
    label: str,
) -> dict:
    values = {task: [value(row) for row in rows[task]] for task in TASKS}
    return {
        "pooled": stratified_summary(values, label=label),
        "per_task_points": {
            task: statistics.mean(values[task]) for task in TASKS
        },
    }


def common_slope_adjustment(rows: dict[str, list[dict]]) -> dict:
    denominator = 0.0
    numerator = 0.0
    task_means = {}
    for task in TASKS:
        xm = statistics.mean(row["delta_mean_length"] for row in rows[task])
        ym = statistics.mean(row["delta_pass1"] for row in rows[task])
        task_means[task] = (xm, ym)
        numerator += sum(
            (row["delta_mean_length"] - xm) * (row["delta_pass1"] - ym)
            for row in rows[task]
        )
        denominator += sum(
            (row["delta_mean_length"] - xm) ** 2 for row in rows[task]
        )
    slope = numerator / denominator if denominator else 0.0
    intercepts = {task: ym - slope * xm for task, (xm, ym) in task_means.items()}
    return {
        "common_slope": slope,
        "task_intercepts_at_zero_length_difference": intercepts,
        "pooled_adjusted_at_zero_length_difference": statistics.mean(intercepts.values()),
    }


def bootstrap_adjusted_intercept(rows: dict[str, list[dict]]) -> dict:
    point = common_slope_adjustment(rows)
    rng = random.Random(f"{BOOTSTRAP_SEED}|length-adjusted-intercept")
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = {
            task: [
                rows[task][rng.randrange(len(rows[task]))] for _ in rows[task]
            ]
            for task in TASKS
        }
        draws.append(
            common_slope_adjustment(sampled)["pooled_adjusted_at_zero_length_difference"]
        )
    draws.sort()
    point.update({"low": percentile(draws, 0.025), "high": percentile(draws, 0.975)})
    return point


def condition_a_robustness(root: Path, queries: dict[str, list[int]]) -> dict:
    rows: dict[str, list[dict]] = {task: [] for task in TASKS}
    for task in TASKS:
        for row in queries[task]:
            ef = load_records(
                [condition_a_record_path(root, task, row, rollout) for rollout in range(64)]
            )
            ao = load_records(
                [phase1_record_path(root, task, "ao_b32", row, rollout) for rollout in range(64)]
            )
            random_records = load_records(
                [
                    phase1_record_path(root, task, "random_b32", row, rollout)
                    for rollout in range(16)
                ]
            )
            ar_records = load_records(
                [
                    phase1_record_path(root, task, "ar_b1", row, rollout)
                    for rollout in range(16)
                ]
            )
            ef_flags = [bool(record["correct"]) for record in ef]
            ao_flags = [bool(record["correct"]) for record in ao]
            random_flags = [bool(record["correct"]) for record in random_records]
            ar_flags = [bool(record["correct"]) for record in ar_records]
            ef_lengths = [len(str(record["response"]).split()) for record in ef]
            ao_lengths = [len(str(record["response"]).split()) for record in ao]
            rows[task].append(
                {
                    "row": row,
                    "ef_successes": sum(ef_flags),
                    "ao_successes": sum(ao_flags),
                    "ef_flags": ef_flags,
                    "ao_flags": ao_flags,
                    "random_flags": random_flags,
                    "ar_flags": ar_flags,
                    "ef_lengths": ef_lengths,
                    "ao_lengths": ao_lengths,
                    "delta_pass1": statistics.mean(ef_flags) - statistics.mean(ao_flags),
                    "delta_mean_length": (
                        statistics.mean(ef_lengths) - statistics.mean(ao_lengths)
                    ),
                }
            )

    grids = {
        "primary_full64_k4_8_16_32_64": (64, PRIMARY_GRID),
        "low_budget_full64_k2_4_8_16": (64, LOW_GRID),
        "high_budget_full64_k16_32_64": (64, HIGH_GRID),
        "frozen_first16_k2_4_8_16": (16, LOW_GRID),
    }
    grid_results = {}
    for name, (n, grid) in grids.items():
        def grid_delta(row: dict, *, n: int = n, grid: tuple[int, ...] = grid) -> float:
            if n == 64:
                ef_successes = row["ef_successes"]
                ao_successes = row["ao_successes"]
            else:
                ef_successes = sum(row["ef_flags"][:n])
                ao_successes = sum(row["ao_flags"][:n])
            return coverage(ef_successes, n, grid) - coverage(ao_successes, n, grid)

        grid_results[name] = contrast_summary(rows, grid_delta, f"grid|{name}")

    per_k = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        per_k[str(k)] = contrast_summary(
            rows,
            lambda row, k=k: (
                pass_at_k(64, row["ef_successes"], k)
                - pass_at_k(64, row["ao_successes"], k)
            ),
            f"per-k|{k}",
        )

    ao_random = {
        "availability": (
            "random_b32 has 16 rollouts only; no matched 64-rollout comparison exists"
        ),
        "matched_first16_per_k": {},
        "asymmetric_ao64_random16_per_k": {},
        "ef_minus_random_matched_first16_per_k": {},
        "ar_minus_random_matched_first16_per_k": {},
    }
    for k in (1, 2, 4, 8, 16):
        ao_random["matched_first16_per_k"][str(k)] = contrast_summary(
            rows,
            lambda row, k=k: (
                pass_at_k(16, sum(row["ao_flags"][:16]), k)
                - pass_at_k(16, sum(row["random_flags"]), k)
            ),
            f"ao-random|matched-first16|{k}",
        )
        # Both estimators target Pass@k, but their Monte Carlo precision differs.
        # This sensitivity analysis must not be described as matched-64 evidence.
        ao_random["asymmetric_ao64_random16_per_k"][str(k)] = contrast_summary(
            rows,
            lambda row, k=k: (
                pass_at_k(64, row["ao_successes"], k)
                - pass_at_k(16, sum(row["random_flags"]), k)
            ),
            f"ao-random|asymmetric-64-16|{k}",
        )
        ao_random["ef_minus_random_matched_first16_per_k"][str(k)] = (
            contrast_summary(
                rows,
                lambda row, k=k: (
                    pass_at_k(16, sum(row["ef_flags"][:16]), k)
                    - pass_at_k(16, sum(row["random_flags"]), k)
                ),
                f"ef-random|matched-first16|{k}",
            )
        )
        ao_random["ar_minus_random_matched_first16_per_k"][str(k)] = (
            contrast_summary(
                rows,
                lambda row, k=k: (
                    pass_at_k(16, sum(row["ar_flags"]), k)
                    - pass_at_k(16, sum(row["random_flags"]), k)
                ),
                f"ar-random|matched-first16|{k}",
            )
        )
    ao_random["matched_first16_low_grid"] = contrast_summary(
        rows,
        lambda row: (
            coverage(sum(row["ao_flags"][:16]), 16, LOW_GRID)
            - coverage(sum(row["random_flags"]), 16, LOW_GRID)
        ),
        "ao-random|matched-first16|low-grid",
    )
    ao_random["asymmetric_ao64_random16_low_grid"] = contrast_summary(
        rows,
        lambda row: (
            coverage(row["ao_successes"], 64, LOW_GRID)
            - coverage(sum(row["random_flags"]), 16, LOW_GRID)
        ),
        "ao-random|asymmetric-64-16|low-grid",
    )
    ao_random["ef_minus_random_matched_first16_low_grid"] = contrast_summary(
        rows,
        lambda row: (
            coverage(sum(row["ef_flags"][:16]), 16, LOW_GRID)
            - coverage(sum(row["random_flags"]), 16, LOW_GRID)
        ),
        "ef-random|matched-first16|low-grid",
    )
    ao_random["ar_minus_random_matched_first16_low_grid"] = contrast_summary(
        rows,
        lambda row: (
            coverage(sum(row["ar_flags"]), 16, LOW_GRID)
            - coverage(sum(row["random_flags"]), 16, LOW_GRID)
        ),
        "ar-random|matched-first16|low-grid",
    )

    length_values = {
        task: [row["delta_pass1"] for row in rows[task]] for task in TASKS
    }
    length_analysis = {
        "raw_ef_minus_ao_pass1": stratified_summary(
            length_values, label="length|raw-pass1"
        ),
        "query_level_associations": {
            task: {
                "spearman_delta_length_delta_pass1": spearman(
                    [row["delta_mean_length"] for row in rows[task]],
                    [row["delta_pass1"] for row in rows[task]],
                ),
                "delta_length_min": min(row["delta_mean_length"] for row in rows[task]),
                "delta_length_max": max(row["delta_mean_length"] for row in rows[task]),
                "delta_length_mean": statistics.mean(
                    row["delta_mean_length"] for row in rows[task]
                ),
            }
            for task in TASKS
        },
        "linear_adjustment": bootstrap_adjusted_intercept(rows),
        "linear_adjustment_warning": (
            "Conditioning on post-treatment decoded length changes the estimand; the "
            "zero-difference intercept may extrapolate beyond common support."
        ),
        "same_seed_near_length_pairs": {},
    }
    for threshold in (2, 5, 10, 20):
        values: dict[str, list[float]] = {task: [] for task in TASKS}
        pair_counts = {}
        for task in TASKS:
            total_pairs = 0
            for row in rows[task]:
                differences = [
                    int(ef_correct) - int(ao_correct)
                    for ef_correct, ao_correct, ef_length, ao_length in zip(
                        row["ef_flags"],
                        row["ao_flags"],
                        row["ef_lengths"],
                        row["ao_lengths"],
                    )
                    if abs(ef_length - ao_length) <= threshold
                ]
                if differences:
                    values[task].append(statistics.mean(differences))
                    total_pairs += len(differences)
            pair_counts[task] = total_pairs
        key = f"absolute_length_difference_le_{threshold}"
        if all(values[task] for task in TASKS):
            length_analysis["same_seed_near_length_pairs"][key] = {
                "summary": stratified_summary(values, label=f"length-matched|{threshold}"),
                "eligible_rollout_pairs": pair_counts,
                "warning": (
                    "Descriptive post-treatment subset of common-seed rollout pairs."
                ),
            }

    return {
        "budget_grid_sensitivity": grid_results,
        "per_k_ef_minus_ao": per_k,
        "ao_minus_random": ao_random,
        "length_conditioned_pass1": length_analysis,
    }


def format_interval(summary: dict) -> str:
    return f"{summary['point']:+.5f} [{summary['low']:+.5f}, {summary['high']:+.5f}]"


def render_markdown(report: dict) -> str:
    phase2 = report["phase2_leverage"]
    condition = report["condition_a_robustness"]
    decomposition = phase2["p2_exact_decomposition"]
    lines = [
        "# Post-Unblinding Robustness Analyses",
        "",
        "**Status:** exploratory, post-unblinding, CPU-only. These analyses do not",
        "alter any frozen endpoint, confidence interval, or decision rule.",
        "",
        "## 1. Phase 2 endpoint leverage",
        "",
        "| Quantity | Estimate |",
        "|---|---:|",
        f"| Mean log2 P2 slope | {decomposition['mean_p2']:+.6f} |",
        f"| B=1-to-parallel step component | {decomposition['b1_to_parallel_step_component']:+.6f} |",
        f"| Within-parallel endpoint component | {decomposition['within_parallel_endpoint_component']:+.6f} |",
        f"| Absolute share attributed to step | {100 * decomposition['absolute_fraction_from_step']:.1f}% |",
        "",
        f"At the aggregate means, `{decomposition['formula']}`. The decomposition",
        "shows that the resolved P2 slope is overwhelmingly leverage from the discrete",
        "B=1-to-block-parallel transition, not evidence of a graded dose response within",
        "the parallel regime. P1 and P2 are therefore dependent views of the same jump,",
        "not independent confirmations.",
        "",
        f"Query-level P1/P2 Pearson correlation: {phase2['p1_p2_dependence']['pearson']:.4f};",
        f"tie-aware Spearman: {phase2['p1_p2_dependence']['spearman_tie_aware']:.4f}.",
        "",
        "### Zero-information and active-subset audit",
        "",
        f"- Zero-success cells: {phase2['zero_information_audit']['zero_cells_of_800']}/800.",
        f"- Queries with all four cells at zero: {phase2['zero_information_audit']['queries_all_four_cells_zero_of_200']}/200.",
        f"- Queries with both frozen P1 endpoint arms at zero: {phase2['zero_information_audit']['queries_b1_and_b32_both_zero_of_200']}/200.",
        f"- Outcome-conditioned active-endpoint P1: {format_interval(phase2['p1_endpoint_active_subset']['summary'])}.",
        "",
        "The active-subset estimate is descriptive selection on observed outcomes and",
        "must never replace the frozen intention-to-treat P1 estimate.",
        "",
        "## 2. Sampling-budget-grid sensitivity",
        "",
        "All `full64` rows below use the same 64 EF and AO rollouts. They isolate the",
        "choice of k-grid from the first-16-versus-64-rollout realization difference.",
        "",
        "| Metric | EF - AO, pooled 95% bootstrap CI |",
        "|---|---:|",
    ]
    for name, result in condition["budget_grid_sensitivity"].items():
        lines.append(f"| `{name}` | {format_interval(result['pooled'])} |")
    lines.extend(
        [
            "",
            "The verdict is a functional of the sampling-budget distribution, not an",
            "intrinsic scalar property of a policy. A first-16 metric additionally mixes",
            "budget weighting with finite-rollout realization and must not be described as",
            "a pure grid effect.",
            "",
            "### Per-k EF - AO contrasts from the same 64 rollouts",
            "",
            "| k | Pooled | GSM8K point | MATH-500 point |",
            "|---:|---:|---:|---:|",
        ]
    )
    for k, result in condition["per_k_ef_minus_ao"].items():
        lines.append(
            f"| {k} | {format_interval(result['pooled'])} | "
            f"{result['per_task_points']['gsm8k']:+.5f} | "
            f"{result['per_task_points']['math500']:+.5f} |"
        )
    ao_random = condition["ao_minus_random"]
    lines.extend(
        [
            "",
            "## 3. AO versus random: matched first-16 decomposition",
            "",
            "`random_b32` has only 16 rollouts. The matched analysis therefore uses",
            "the common first 16 AO and random rollouts; no matched-64 result exists.",
            "",
            "| k | AO - random, matched first 16 | GSM8K point | MATH-500 point |",
            "|---:|---:|---:|---:|",
        ]
    )
    for k, result in ao_random["matched_first16_per_k"].items():
        lines.append(
            f"| {k} | {format_interval(result['pooled'])} | "
            f"{result['per_task_points']['gsm8k']:+.5f} | "
            f"{result['per_task_points']['math500']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "Matched low-grid AO-random aggregate: "
            f"{format_interval(ao_random['matched_first16_low_grid']['pooled'])}.",
            "The JSON also records an AO-64-versus-random-16 estimator-precision",
            "sensitivity analysis. It is not matched-64 evidence and is excluded from",
            "the headline table.",
            "",
            "### Matched first-16 checks against random",
            "",
            "| k | EF - random | AR - random |",
            "|---:|---:|---:|",
        ]
    )
    for k in ao_random["ef_minus_random_matched_first16_per_k"]:
        ef_result = ao_random["ef_minus_random_matched_first16_per_k"][k]
        ar_result = ao_random["ar_minus_random_matched_first16_per_k"][k]
        lines.append(
            f"| {k} | {format_interval(ef_result['pooled'])} | "
            f"{format_interval(ar_result['pooled'])} |"
        )
    lines.extend(
        [
            "",
            "These comparisons use the same 16 rollout indices in each policy. They",
            "replace asymmetric point-estimate comparisons when assessing dominance.",
        ]
    )
    length = condition["length_conditioned_pass1"]
    adjusted = length["linear_adjustment"]
    lines.extend(
        [
            "",
            "## 4. Length-conditioned pass@1 diagnostics",
            "",
            f"Raw EF-AO pass@1: {format_interval(length['raw_ef_minus_ao_pass1'])}.",
            f"A task-fixed-intercept linear adjustment evaluated at zero length difference: "
            f"{adjusted['pooled_adjusted_at_zero_length_difference']:+.5f} "
            f"[{adjusted['low']:+.5f}, {adjusted['high']:+.5f}].",
            "",
            "This adjustment is not causal: decoded length is post-treatment, and zero",
            "length difference may be outside common support. Same-seed near-length subsets",
            "provide a complementary descriptive check:",
            "",
            "| Pair filter | EF - AO correctness | Eligible pairs (GSM8K / MATH-500) | Queries |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in length["same_seed_near_length_pairs"].items():
        summary = result["summary"]
        pairs = result["eligible_rollout_pairs"]
        lines.append(
            f"| `{name}` | {format_interval(summary)} | "
            f"{pairs['gsm8k']} / {pairs['math500']} | "
            f"{summary['n_gsm8k']} / {summary['n_math500']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "1. The Phase 2 result supports a sequential-versus-block-parallel threshold",
            "   effect. It does not establish a smooth freedom-dose law.",
            "2. Coverage summaries must declare their k-grid or sampling-budget weighting;",
            "   crossing frontiers cannot be represented faithfully by one universal AUC.",
            "3. Length-conditioned analyses are exploratory mediator-conditioned",
            "   descriptions, not repairs to the frozen Condition A endpoint.",
            "4. AO-random, EF-random, and AR-random contrasts share queries and the",
            "   random control; they are complementary instances, not independent replications.",
            "5. No new GPU inference was used.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _, queries = load_manifest(args.root)
    report = {
        "analysis_status": "exploratory_post_unblinding_cpu_only",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "phase2_leverage": phase2_leverage(args.root, queries),
        "condition_a_robustness": condition_a_robustness(args.root, queries),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "posthoc_robustness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "posthoc_robustness.md").write_text(render_markdown(report))


if __name__ == "__main__":
    main()
