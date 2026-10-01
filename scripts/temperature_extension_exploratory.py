"""Post-outcome, CPU-only temperature interaction and trace diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from dllm_order_transmission.artifacts import write_json_atomic
from dllm_order_transmission.phase1_protocol import load_split_manifest
from dllm_order_transmission.temperature_extension_analyzer import (
    TASKS,
    _control_result_path,
    _high,
    _new_result_path,
    _pass,
    _queries,
    _trace_path,
)


SCHEDULES = ("sequential", "confidence_first", "random")
TEMPERATURES = ("0.6", "0.9", "1.2")
NEW_POLICY = {
    ("sequential", "0.9"): "sequential_b1_t09",
    ("confidence_first", "0.9"): "cf_b32_t09",
    ("random", "0.9"): "rnd_b32_t09",
    ("sequential", "1.2"): "sequential_b1_t12",
    ("confidence_first", "1.2"): "cf_b32_t12",
    ("random", "1.2"): "rnd_b32_t12",
}
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPLICATES = 10_000
TRACE_QUERIES_PER_TASK = 100
TRACE_ROLLOUTS = (0, 1)


def result_path(
    phase1_root: Path,
    temperature_root: Path,
    task: str,
    row: int,
    schedule: str,
    temperature: str,
    rollout: int,
) -> Path:
    if temperature == "0.6":
        return _control_result_path(phase1_root, task, row, schedule, rollout)
    return _new_result_path(
        temperature_root,
        "temperature-formal",
        task,
        NEW_POLICY[(schedule, temperature)],
        row,
        rollout,
    )


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def paired_bootstrap(values: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    names = tuple(values)
    points = {
        name: statistics.mean(statistics.mean(values[name][task]) for task in TASKS)
        for name in names
    }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = {name: [] for name in names}
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = {
            task: [rng.randrange(len(values[names[0]][task])) for _ in values[names[0]][task]]
            for task in TASKS
        }
        for name in names:
            draws[name].append(
                statistics.mean(
                    statistics.mean(values[name][task][i] for i in indices[task])
                    for task in TASKS
                )
            )
    return {
        name: {
            "point": points[name],
            "raw_95_ci": [percentile(draws[name], 0.025), percentile(draws[name], 0.975)],
            "per_task_points": {
                task: statistics.mean(values[name][task]) for task in TASKS
            },
        }
        for name in names
    }


def trace_means(path: Path) -> dict[str, float]:
    selected_entropy = 0.0
    eligible_entropy = 0.0
    steps = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            eligible = event["eligible_positions"]
            selected = event["selected_positions"]
            entropy = event["entropy"]
            if len(selected) != 1 or len(eligible) != len(entropy):
                raise ValueError(f"unexpected trace event shape in {path}")
            index = eligible.index(selected[0])
            selected_entropy += entropy[index]
            eligible_entropy += statistics.mean(entropy)
            steps += 1
    if steps != 256:
        raise ValueError(f"expected 256 trace events in {path}, got {steps}")
    means = {
        "selected_entropy": selected_entropy / steps,
        "eligible_entropy": eligible_entropy / steps,
    }
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError(f"nonfinite trace diagnostic in {path}")
    return means


def run(
    *, phase1_root: Path, temperature_root: Path, manifest_path: Path
) -> dict[str, Any]:
    queries = _queries(load_split_manifest(manifest_path), "heldout")
    metric_by_cell: dict[tuple[str, int, str, str], dict[str, float]] = {}
    for task in TASKS:
        for query in queries[task]:
            row = int(query["row_index"])
            for temperature in TEMPERATURES:
                for schedule in SCHEDULES:
                    flags = [
                        bool(
                            json.loads(
                                result_path(
                                    phase1_root,
                                    temperature_root,
                                    task,
                                    row,
                                    schedule,
                                    temperature,
                                    rollout,
                                ).read_text(encoding="utf-8")
                            )["correct"]
                        )
                        for rollout in range(32)
                    ]
                    metric_by_cell[(task, row, schedule, temperature)] = {
                        "low": _pass(flags, 1),
                        "high": _high(flags),
                    }

    values: dict[str, dict[str, list[float]]] = {}
    for temperature in TEMPERATURES[1:]:
        for endpoint in ("low", "high"):
            for schedule in SCHEDULES:
                name = f"{schedule}_t{temperature}_minus_t0.6_{endpoint}"
                values[name] = {task: [] for task in TASKS}
            interaction = f"random_vs_cf_t{temperature}_minus_t0.6_{endpoint}"
            values[interaction] = {task: [] for task in TASKS}
            for task in TASKS:
                for query in queries[task]:
                    row = int(query["row_index"])
                    changes = {
                        schedule: (
                            metric_by_cell[(task, row, schedule, temperature)][endpoint]
                            - metric_by_cell[(task, row, schedule, "0.6")][endpoint]
                        )
                        for schedule in SCHEDULES
                    }
                    for schedule in SCHEDULES:
                        values[f"{schedule}_t{temperature}_minus_t0.6_{endpoint}"][task].append(
                            changes[schedule]
                        )
                    values[interaction][task].append(
                        changes["random"] - changes["confidence_first"]
                    )

    trace_cells: dict[str, Any] = {}
    selected_rows: dict[str, list[int]] = {}
    for task in TASKS:
        task_queries = queries[task]
        if len(task_queries) != 100:
            raise ValueError(f"expected 100 held-out queries for {task}")
        selected_rows[task] = [
            int(task_queries[index]["row_index"])
            for index in range(0, len(task_queries), len(task_queries) // TRACE_QUERIES_PER_TASK)
        ][:TRACE_QUERIES_PER_TASK]
    for temperature in TEMPERATURES:
        trace_cells[temperature] = {}
        for schedule in SCHEDULES:
            per_trace = []
            for task in TASKS:
                for row in selected_rows[task]:
                    for rollout in TRACE_ROLLOUTS:
                        per_trace.append(
                            trace_means(
                                _trace_path(
                                    result_path(
                                        phase1_root,
                                        temperature_root,
                                        task,
                                        row,
                                        schedule,
                                        temperature,
                                        rollout,
                                    )
                                )
                            )
                        )
            trace_cells[temperature][schedule] = {
                "trace_count": len(per_trace),
                **{
                    name: statistics.mean(trace[name] for trace in per_trace)
                    for name in per_trace[0]
                },
            }

    return {
        "evidence_status": "post-outcome exploratory diagnostic; cannot revise frozen verdict",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_intervals": "unadjusted percentile 95% CI, query-paired and task-stratified",
        "interactions": paired_bootstrap(values),
        "trace_sample": {
            "selection_rule": "all held-out queries in sorted manifest order, rollouts 0 and 1",
            "rows": selected_rows,
            "rollouts": list(TRACE_ROLLOUTS),
            "cells": trace_cells,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--temperature-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        phase1_root=args.phase1_root,
        temperature_root=args.temperature_root,
        manifest_path=args.split_manifest,
    )
    write_json_atomic(args.output_json, report)
    print(json.dumps({"interactions": report["interactions"], "trace_cells": report["trace_sample"]["cells"]}, indent=2))


if __name__ == "__main__":
    main()
