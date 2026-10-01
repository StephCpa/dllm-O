#!/usr/bin/env python3
"""Query-level post-review audits over the completed held-out artifacts.

These analyses are exploratory and do not replace any frozen endpoint.  The
script deliberately uses the same query-level bootstrap convention as the
existing post-review audit, while adding effect distributions, influence,
dispersion diagnostics, majority-vote intervals, and same-policy A/A checks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from postreview_cpu_audits import (  # noqa: E402
    K64,
    POLICIES,
    TASKS,
    load_all,
    load_queries,
    majority_vote_correct,
    pass_at_k,
)


BOOTSTRAP_SEED = 20261001
BOOTSTRAP_REPLICATES = 10_000
PRIMARY_GRID = (4, 8, 16, 32, 64)


def paired_bootstrap(
    values: dict[str, list[float]], label: str, seed_offset: int = 0
) -> dict[str, float]:
    """Return a task-balanced query bootstrap interval for paired values."""
    point = statistics.mean(statistics.mean(values[task]) for task in TASKS)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        task_means = []
        for task in TASKS:
            source = np.asarray(values[task], dtype=float)
            indices = rng.integers(0, len(source), size=len(source))
            task_means.append(float(source[indices].mean()))
        draws.append(statistics.mean(task_means))
    return {
        "label": label,
        "point": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def contrast_value(records, task: str, row: int, left: str, right: str, grid) -> float:
    left_cell = records[(task, row, left)]
    right_cell = records[(task, row, right)]
    return statistics.mean(
        pass_at_k(64, sum(r["correct"] for r in left_cell), k)
        - pass_at_k(64, sum(r["correct"] for r in right_cell), k)
        for k in grid
    )


def effect_distribution(records, queries) -> dict:
    values = {task: [] for task in TASKS}
    rows = []
    for task in TASKS:
        for row in queries[task]:
            effect = contrast_value(records, task, row, "ao_b32", "ar_b1", PRIMARY_GRID)
            values[task].append(effect)
            rows.append({"task": task, "row": row, "effect": effect})
    flat = [item["effect"] for item in rows]
    absolute = sorted(rows, key=lambda item: abs(item["effect"]), reverse=True)
    return {
        "grid": list(PRIMARY_GRID),
        "summary": {
            "n_queries": len(flat),
            "mean": statistics.mean(flat),
            "median": statistics.median(flat),
            "q10": float(np.quantile(flat, 0.10)),
            "q25": float(np.quantile(flat, 0.25)),
            "q75": float(np.quantile(flat, 0.75)),
            "q90": float(np.quantile(flat, 0.90)),
            "positive": sum(value > 1e-12 for value in flat),
            "negative": sum(value < -1e-12 for value in flat),
            "zero": sum(abs(value) <= 1e-12 for value in flat),
        },
        "per_task_bootstrap": paired_bootstrap(values, "ao_b32-minus-ar_b1-primary"),
        "largest_absolute_effects": absolute[:10],
        "task_values": values,
    }


def leave_one_query_out(records, queries) -> dict:
    values = {
        task: [contrast_value(records, task, row, "ao_b32", "ar_b1", PRIMARY_GRID) for row in queries[task]]
        for task in TASKS
    }
    task_means = {task: statistics.mean(values[task]) for task in TASKS}
    full = statistics.mean(task_means.values())
    influences = []
    for task in TASKS:
        other_task = next(name for name in TASKS if name != task)
        for index, row in enumerate(queries[task]):
            loo_task_mean = (sum(values[task]) - values[task][index]) / (len(values[task]) - 1)
            loo = statistics.mean((loo_task_mean, task_means[other_task]))
            influences.append(
                {
                    "task": task,
                    "row": row,
                    "effect": values[task][index],
                    "full_minus_loo": full - loo,
                }
            )
    ranked = sorted(influences, key=lambda item: abs(item["full_minus_loo"]), reverse=True)
    return {
        "full_point": full,
        "max_abs_full_minus_loo": max(abs(item["full_minus_loo"]) for item in influences),
        "top_influences": ranked[:10],
    }


def dispersion(records, queries) -> dict:
    result = {}
    for task in TASKS:
        result[task] = {}
        for policy in ("ar_b1", "ao_b32", "random_b32", "ef_b32"):
            counts = np.asarray(
                [sum(record["correct"] for record in records[(task, row, policy)]) for row in queries[task]],
                dtype=float,
            )
            proportions = counts / 64.0
            mean_p = float(proportions.mean())
            sample_var = float(proportions.var(ddof=1))
            binomial_var = mean_p * (1.0 - mean_p) / 64.0
            denominator = 64.0 * 63.0 * mean_p * (1.0 - mean_p)
            rho = (float(counts.var(ddof=1)) - 64.0 * mean_p * (1.0 - mean_p)) / denominator if denominator else 0.0
            result[task][policy] = {
                "n_queries": len(counts),
                "mean_success_fraction": mean_p,
                "sample_variance_success_fraction": sample_var,
                "binomial_reference_variance": binomial_var,
                "variance_ratio_to_binomial_reference": sample_var / binomial_var if binomial_var else None,
                "beta_binomial_moment_rho": rho,
                "zero_success_queries": int(np.sum(counts == 0)),
            }
    return result


def majority_vote_intervals(records, queries) -> dict:
    result = {}
    for k in (1, 16, 32, 64):
        result[str(k)] = {}
        for left, right in (("ar_b1", "random_b32"), ("ao_b32", "random_b32")):
            values = {task: [] for task in TASKS}
            for task in TASKS:
                for row in queries[task]:
                    left_value = majority_vote_correct(records[(task, row, left)][:k])
                    right_value = majority_vote_correct(records[(task, row, right)][:k])
                    values[task].append(left_value - right_value)
            result[str(k)][f"{left}_minus_{right}"] = paired_bootstrap(
                values, f"majority-{left}-minus-{right}-k{k}", seed_offset=k
            )
    return result


def aa_split(records, queries) -> dict:
    result = {}
    for policy in ("ar_b1", "ao_b32", "random_b32"):
        result[policy] = {}
        for k in (1, 4, 8, 16):
            values = {task: [] for task in TASKS}
            for task in TASKS:
                for row in queries[task]:
                    cell = records[(task, row, policy)]
                    values[task].append(
                        pass_at_k(32, sum(r["correct"] for r in cell[:32]), k)
                        - pass_at_k(32, sum(r["correct"] for r in cell[32:]), k)
                    )
            result[policy][str(k)] = paired_bootstrap(values, f"aa-{policy}-k{k}", seed_offset=100 + k)
    return result


def fmt(item: dict) -> str:
    return f"{item['point']:+.4f} [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]"


def render(report: dict) -> str:
    dist = report["effect_distribution"]
    summary = dist["summary"]
    lines = [
        "# Query-Level Exploratory Audits",
        "",
        "**Status:** post-outcome exploratory, CPU-only. These analyses do not",
        "replace or reinterpret any frozen endpoint.",
        "",
        f"Bootstrap seed: `{BOOTSTRAP_SEED}`; replicates: `{BOOTSTRAP_REPLICATES}`.",
        "",
        "## Block-size effect distribution",
        "",
        "For each held-out query, this report averages $B=32-B=1$ over "
        "$k\\in\\{4,8,16,32,64\\}$. The pooled query-bootstrap estimate is "
        f"{fmt(dist['per_task_bootstrap'])}.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| queries | {summary['n_queries']} |",
        f"| mean / median | {summary['mean']:+.4f} / {summary['median']:+.4f} |",
        f"| q10 / q25 | {summary['q10']:+.4f} / {summary['q25']:+.4f} |",
        f"| q75 / q90 | {summary['q75']:+.4f} / {summary['q90']:+.4f} |",
        f"| positive / negative / zero | {summary['positive']} / {summary['negative']} / {summary['zero']} |",
        "",
        "This describes heterogeneity; it is not a query-level significance test.",
        "",
        "## Influence",
        "",
        f"The largest absolute leave-one-query-out change is "
        f"`{dist['summary']['n_queries'] and report['leave_one_query_out']['max_abs_full_minus_loo']:.6f}` "
        f"on the pooled primary-grid estimate. The ten largest cases are retained "
        "in JSON for audit, but are not treated as outlier exclusions.",
        "",
        "## Dispersion diagnostic",
        "",
        "The variance ratio compares between-query success-fraction variance with "
        "the binomial reference at 64 rollouts. The beta-binomial moment rho is "
        "descriptive only; no beta-binomial null is fitted or used for inference.",
        "",
        "| task | policy | mean success | variance ratio | rho | zero-success queries |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for policy in ("ar_b1", "ao_b32", "random_b32", "ef_b32"):
            item = report["dispersion"][task][policy]
            ratio = item["variance_ratio_to_binomial_reference"]
            ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
            lines.append(
                f"| {task} | {policy} | {item['mean_success_fraction']:.3f} | "
                f"{ratio_text} | {item['beta_binomial_moment_rho']:+.4f} | "
                f"{item['zero_success_queries']} |"
            )
    lines.extend(
        [
            "",
            "## Majority-vote paired intervals",
            "",
            "Unadjusted query-bootstrap intervals for descriptive, tie-aware majority-vote differences:",
            "",
            "| k | Sequential - RND | CF - RND |",
            "|---:|---:|---:|",
        ]
    )
    for k in (1, 16, 32, 64):
        item = report["majority_vote_intervals"][str(k)]
        lines.append(
            f"| {k} | {fmt(item['ar_b1_minus_random_b32'])} | "
            f"{fmt(item['ao_b32_minus_random_b32'])} |"
        )
    lines.extend(
        [
            "",
            "## Same-policy A/A split",
            "",
            "Each policy is split into rollout indices 0--31 versus 32--63. These are",
            "calibration diagnostics, not independent replication or evidence of a",
            "policy effect.",
            "",
            "| policy | k=1 | k=4 | k=8 | k=16 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in ("ar_b1", "ao_b32", "random_b32"):
        lines.append(
            f"| {policy} | "
            + " | ".join(fmt(report["aa_split"][policy][str(k)]) for k in (1, 4, 8, 16))
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    queries = load_queries(args.root)
    records = load_all(args.root, queries)
    report = {
        "analysis_status": "post_outcome_exploratory_cpu_only",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "effect_distribution": effect_distribution(records, queries),
        "leave_one_query_out": leave_one_query_out(records, queries),
        "dispersion": dispersion(records, queries),
        "majority_vote_intervals": majority_vote_intervals(records, queries),
        "aa_split": aa_split(records, queries),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "query_level_exploratory_audits.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "query_level_exploratory_audits.md").write_text(
        render(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
