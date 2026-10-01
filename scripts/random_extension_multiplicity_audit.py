#!/usr/bin/env python3
"""Multiplicity audit for the post-unblinding random-policy extension.

This analysis is exploratory. It cannot alter the frozen extension verdict.
It asks whether the random-minus-AO sign reversal survives simultaneous
inference over the seven declared per-k secondary endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


TASKS = ("gsm8k", "math500")
K_GRID = (1, 2, 4, 8, 16, 32, 64)
LOW_INDICES = (0, 1)
HIGH_INDICES = (4, 5)
DEFAULT_SEED = 20260914
DEFAULT_REPLICATES = 50_000


def pass_at_k(n: int, successes: int, k: int) -> float:
    if n - successes < k:
        return 1.0
    return 1.0 - math.comb(n - successes, k) / math.comb(n, k)


def _record_path(
    root: Path, task: str, policy: str, row: int, rollout: int
) -> Path:
    if policy == "random_b32" and rollout >= 16:
        run_kind = "random-extension"
    else:
        run_kind = "screening" if rollout < 16 else "primary-extension"
    return (
        root
        / "phase1-v1"
        / "heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def load_differences(
    phase1_root: Path, split_manifest: Path
) -> dict[str, np.ndarray]:
    manifest = json.loads(split_manifest.read_text())
    differences = {}
    for task in TASKS:
        task_rows = []
        entries = sorted(
            manifest["splits"][task]["heldout"], key=lambda row: row["row_index"]
        )
        for entry in entries:
            row = int(entry["row_index"])
            counts = {}
            for policy in ("random_b32", "ao_b32"):
                counts[policy] = sum(
                    bool(
                        json.loads(
                            _record_path(
                                phase1_root, task, policy, row, rollout
                            ).read_text()
                        )["correct"]
                    )
                    for rollout in range(64)
                )
            task_rows.append(
                [
                    pass_at_k(64, counts["random_b32"], k)
                    - pass_at_k(64, counts["ao_b32"], k)
                    for k in K_GRID
                ]
            )
        differences[task] = np.asarray(task_rows, dtype=float)
    return differences


def stratified_bootstrap(
    differences: dict[str, np.ndarray], *, seed: int, replicates: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    point = np.mean(
        np.stack([differences[task].mean(axis=0) for task in TASKS]), axis=0
    )
    draws = np.empty((replicates, len(K_GRID)), dtype=float)
    batch_size = 1_000
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        task_draws = []
        for task in TASKS:
            values = differences[task]
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            task_draws.append(values[indices].mean(axis=1))
        draws[start:stop] = np.mean(np.stack(task_draws), axis=0)
    return point, draws


def interval(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def simultaneous_bands(
    point: np.ndarray, draws: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, float]:
    critical = float(np.quantile(np.max(np.abs(draws - point), axis=1), 1 - alpha))
    return point - critical, point + critical, critical


def summarize(
    differences: dict[str, np.ndarray], *, seed: int, replicates: int
) -> dict:
    point, draws = stratified_bootstrap(
        differences, seed=seed, replicates=replicates
    )
    simultaneous_low, simultaneous_high, critical = simultaneous_bands(point, draws)
    bonferroni_alpha = 0.05 / len(K_GRID)

    per_k = {}
    for index, k in enumerate(K_GRID):
        raw_low, raw_high = interval(draws[:, index])
        bonf_low, bonf_high = interval(draws[:, index], bonferroni_alpha)
        per_k[str(k)] = {
            "point": float(point[index]),
            "raw_95_ci": [raw_low, raw_high],
            "bonferroni_95_family_ci": [bonf_low, bonf_high],
            "max_deviation_95_simultaneous_ci": [
                float(simultaneous_low[index]),
                float(simultaneous_high[index]),
            ],
        }

    low_point = float(point[list(LOW_INDICES)].mean())
    high_point = float(point[list(HIGH_INDICES)].mean())
    low_draws = draws[:, list(LOW_INDICES)].mean(axis=1)
    high_draws = draws[:, list(HIGH_INDICES)].mean(axis=1)
    joint_point = np.asarray([low_point, high_point])
    joint_draws = np.column_stack([low_draws, high_draws])
    joint_low, joint_high, joint_critical = simultaneous_bands(
        joint_point, joint_draws
    )
    reversal_draws = high_draws - low_draws
    reversal_low, reversal_high = interval(reversal_draws)

    return {
        "status": "post-outcome exploratory multiplicity audit",
        "seed": seed,
        "bootstrap_replicates": replicates,
        "family": {
            "contrast": "random_b32 - ao_b32",
            "k_grid": list(K_GRID),
            "endpoint_count": len(K_GRID),
            "method": "task-stratified paired query bootstrap",
            "simultaneous_method": "nonstudentized maximum absolute deviation",
            "simultaneous_critical_value": critical,
        },
        "per_k": per_k,
        "joint_low_high_summary": {
            "definition": {
                "low": "mean random-minus-AO Pass@k over k={1,2}",
                "high": "mean random-minus-AO Pass@k over k={16,32}",
            },
            "low_point": low_point,
            "low_simultaneous_ci": [float(joint_low[0]), float(joint_high[0])],
            "high_point": high_point,
            "high_simultaneous_ci": [float(joint_low[1]), float(joint_high[1])],
            "simultaneous_critical_value": joint_critical,
            "joint_reversal_resolved": bool(
                joint_high[0] < 0 and joint_low[1] > 0
            ),
            "high_minus_low": {
                "point": high_point - low_point,
                "raw_95_ci": [reversal_low, reversal_high],
            },
        },
        "interpretation_boundary": (
            "This audit was designed after the extension outcomes were read. "
            "It diagnoses multiplicity sensitivity and cannot replace or "
            "strengthen the frozen primary verdict."
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Random-AO Per-k Multiplicity Audit",
        "",
        "**Status:** post-outcome exploratory; does not alter the frozen primary verdict.",
        "",
        "| k | Point | Raw 95% CI | Bonferroni family CI | Simultaneous CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, row in report["per_k"].items():
        raw = row["raw_95_ci"]
        bonf = row["bonferroni_95_family_ci"]
        simultaneous = row["max_deviation_95_simultaneous_ci"]
        lines.append(
            f"| {k} | {row['point']:+.5f} | [{raw[0]:+.5f}, {raw[1]:+.5f}] "
            f"| [{bonf[0]:+.5f}, {bonf[1]:+.5f}] "
            f"| [{simultaneous[0]:+.5f}, {simultaneous[1]:+.5f}] |"
        )
    joint = report["joint_low_high_summary"]
    lines.extend(
        [
            "",
            "## Exploratory joint reversal summary",
            "",
            f"- Low-k mean: {joint['low_point']:+.5f}, simultaneous CI "
            f"[{joint['low_simultaneous_ci'][0]:+.5f}, "
            f"{joint['low_simultaneous_ci'][1]:+.5f}].",
            f"- High-k mean: {joint['high_point']:+.5f}, simultaneous CI "
            f"[{joint['high_simultaneous_ci'][0]:+.5f}, "
            f"{joint['high_simultaneous_ci'][1]:+.5f}].",
            f"- Joint reversal resolved: `{joint['joint_reversal_resolved']}`.",
            "",
            report["interpretation_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    differences = load_differences(args.phase1_root, args.split_manifest)
    report = summarize(differences, seed=args.seed, replicates=args.replicates)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(report))
    print(json.dumps(report["joint_low_high_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
