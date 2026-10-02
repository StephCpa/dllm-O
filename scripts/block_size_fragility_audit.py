#!/usr/bin/env python3
"""Fragility audit for the frozen block-size primary contrast.

This is a post-outcome, CPU-only description of how concentrated the frozen
``CoverageAUC(B=32) - CoverageAUC(B=1)`` effect is across held-out queries.
It reads the per-query effects already committed by
``query_level_exploratory_audits.py`` and repeatedly removes the queries with
the most negative effects, recomputing the pooled estimate and its
task-stratified query-bootstrap interval after each removal.

Removing queries in order of their observed effect is outcome-conditioned by
construction. The result describes how many queries carry the effect; it is
not a significance test, an outlier rule, or a replacement for the frozen
endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 20261002
BOOTSTRAP_REPLICATES = 10_000
MAX_REMOVED = 40
TASKS = ("gsm8k", "math500")
CONCENTRATION_POINTS = (1, 5, 10, 20, 40)


def pooled_point(values: dict[str, np.ndarray]) -> float:
    """Equal-task mean of per-task query means, as in the frozen analysis."""
    return float(np.mean([values[task].mean() for task in TASKS]))


def pooled_bootstrap(values: dict[str, np.ndarray], seed: int) -> dict[str, float]:
    """Task-stratified percentile bootstrap over queries for the pooled mean."""
    rng = np.random.default_rng(seed)
    task_means = []
    for task in TASKS:
        source = values[task]
        indices = rng.integers(0, len(source), size=(BOOTSTRAP_REPLICATES, len(source)))
        task_means.append(source[indices].mean(axis=1))
    draws = np.mean(task_means, axis=0)
    return {
        "point": pooled_point(values),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def removal_order(values: dict[str, np.ndarray], most_negative_first: bool) -> list[tuple[str, int, float]]:
    """Rank every query across both tasks by its observed effect."""
    rows = [(task, index, float(effect)) for task in TASKS for index, effect in enumerate(values[task])]
    # Ties are broken by task and index so the order is deterministic.
    return sorted(rows, key=lambda row: (row[2] if most_negative_first else -row[2], row[0], row[1]))


def remove(values: dict[str, np.ndarray], removed: list[tuple[str, int, float]]) -> dict[str, np.ndarray]:
    dropped = {task: {index for name, index, _ in removed if name == task} for task in TASKS}
    return {
        task: np.asarray([v for i, v in enumerate(values[task]) if i not in dropped[task]], dtype=float)
        for task in TASKS
    }


def removal_curve(values: dict[str, np.ndarray], most_negative_first: bool, max_removed: int) -> list[dict]:
    order = removal_order(values, most_negative_first)
    curve = []
    for m in range(max_removed + 1):
        remaining = remove(values, order[:m])
        interval = pooled_bootstrap(remaining, BOOTSTRAP_SEED + m)
        curve.append(
            {
                "removed": m,
                "removed_gsm8k": sum(1 for task, _, _ in order[:m] if task == "gsm8k"),
                "removed_math500": sum(1 for task, _, _ in order[:m] if task == "math500"),
                "last_removed_effect": order[m - 1][2] if m else None,
                **interval,
            }
        )
    return curve


def first_index(curve: list[dict], condition) -> int | None:
    return next((row["removed"] for row in curve if condition(row)), None)


def concentration(values: dict[str, np.ndarray]) -> dict:
    """Share of the summed per-query effect carried by the most negative queries.

    Shares use the plain sum over all 200 queries. Because both tasks have 100
    queries, this sum is proportional to the equal-task pooled estimate.
    """
    flat = np.sort(np.concatenate([values[task] for task in TASKS]))
    total = float(flat.sum())
    return {
        "total_sum": total,
        "share_by_most_negative": {str(m): float(flat[:m].sum() / total) for m in CONCENTRATION_POINTS},
        "abs_effect_below_0_01": int(np.sum(np.abs(flat) < 0.01)),
        "effect_below_minus_0_05": int(np.sum(flat < -0.05)),
        "effect_above_plus_0_05": int(np.sum(flat > 0.05)),
    }


def audit(task_values: dict[str, list[float]], max_removed: int = MAX_REMOVED) -> dict:
    values = {task: np.asarray(task_values[task], dtype=float) for task in TASKS}
    hurt_first = removal_curve(values, most_negative_first=True, max_removed=max_removed)
    helped_first = removal_curve(values, most_negative_first=False, max_removed=max_removed)
    return {
        "analysis_status": "post_outcome_exploratory_cpu_only",
        "contrast": "CoverageAUC_K64(ao_b32) - CoverageAUC_K64(ar_b1)",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "n_queries": {task: int(len(values[task])) for task in TASKS},
        "full": hurt_first[0],
        "concentration": concentration(values),
        "most_negative_removed_first": {
            "first_removed_where_interval_reaches_zero": first_index(hurt_first, lambda r: r["ci_high"] >= 0),
            "first_removed_where_point_is_nonnegative": first_index(hurt_first, lambda r: r["point"] >= 0),
            "curve": hurt_first,
        },
        "most_positive_removed_first": {
            "note": "Reference direction: removing the queries that favor B=32 strengthens the contrast.",
            "curve": helped_first,
        },
    }


def fmt(row: dict) -> str:
    return f"{row['point']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def render(report: dict) -> str:
    hurt = report["most_negative_removed_first"]
    conc = report["concentration"]
    m_ci = hurt["first_removed_where_interval_reaches_zero"]
    m_point = hurt["first_removed_where_point_is_nonnegative"]
    lines = [
        "# Block-Size Primary: Fragility Audit",
        "",
        "**Status:** post-outcome exploratory, CPU-only. Queries are removed in",
        "order of their observed effect, so this describes how concentrated the",
        "frozen effect is. It is not a test, an outlier rule, or a replacement for",
        "the frozen endpoint.",
        "",
        f"Contrast: `{report['contrast']}`. Bootstrap: task-stratified percentile,",
        f"{report['bootstrap_replicates']:,} replicates, seed `{report['bootstrap_seed']}` (+ number removed).",
        "",
        f"Full panel ({sum(report['n_queries'].values())} queries): {fmt(report['full'])}.",
        "",
        "## Concentration",
        "",
        f"- Queries with |effect| < 0.01: {conc['abs_effect_below_0_01']} of 200.",
        f"- Queries with effect < -0.05: {conc['effect_below_minus_0_05']}; with effect > +0.05: "
        f"{conc['effect_above_plus_0_05']}.",
        "",
        "| most negative queries | share of the summed effect |",
        "|---:|---:|",
    ]
    for m, share in conc["share_by_most_negative"].items():
        lines.append(f"| {m} | {share:.2f} |")
    lines += [
        "",
        "Shares above 1 mean those queries carry more than the net effect, which",
        "queries favoring $B=32$ partly offset.",
        "",
        "## Removing the most negative queries first",
        "",
        f"- The interval first reaches zero after removing **{m_ci}** queries.",
        f"- The point estimate first becomes nonnegative after removing **{m_point}** queries.",
        "",
        "| removed | GSM8K / MATH-500 removed | last removed effect | pooled estimate [95% CI] |",
        "|---:|---:|---:|---:|",
    ]
    for row in hurt["curve"]:
        if row["removed"] <= 25 or row["removed"] % 5 == 0:
            last = "" if row["last_removed_effect"] is None else f"{row['last_removed_effect']:+.4f}"
            lines.append(
                f"| {row['removed']} | {row['removed_gsm8k']} / {row['removed_math500']} | {last} | {fmt(row)} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A heavy-tailed effect is expected to be fragile under outcome-selected",
        "removal, so this number does not by itself weaken the frozen interval,",
        "which already accounts for query-level sampling variability. It does",
        "show that the effect is carried by a small subset of the 200 queries.",
        "Therefore the most informative robustness checks are new queries or a",
        "second model, not further reanalysis of this panel.",
        "",
        "Once enough negative queries are removed, the remaining contrast becomes",
        "positive and eventually resolves above zero. This is produced by the",
        "selection itself: dropping the most negative values always raises the",
        "mean. It must not be read as evidence that $B=32$ helps most queries.",
        "The opposite reference curve (removing the most positive queries first)",
        "is retained in the JSON.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--query-audit",
        type=Path,
        default=Path("reports/query_level_exploratory_audits_20261001/query_level_exploratory_audits.json"),
        help="Committed query-level audit JSON with per-query effects.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/block_size_fragility_audit_20261002"),
    )
    args = parser.parse_args()
    source = json.loads(args.query_audit.read_text(encoding="utf-8"))
    distribution = source["effect_distribution"]
    report = audit(distribution["task_values"])
    expected = distribution["per_task_bootstrap"]["point"]
    if abs(report["full"]["point"] - expected) > 1e-12:
        raise ValueError("Per-query effects do not reproduce the committed pooled estimate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "block_size_fragility_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "block_size_fragility_audit.md").write_text(render(report), encoding="utf-8")


if __name__ == "__main__":
    main()
