#!/usr/bin/env python3
"""Post-review exploratory audits over completed held-out generations.

This script never changes a frozen endpoint or verdict. It addresses reviewer
questions that became visible only after unblinding and labels every output as
post-outcome exploratory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


TASKS = ("gsm8k", "math500")
POLICIES = ("ar_b1", "ao_b8", "ao_b16", "ao_b32", "random_b32", "ef_b32")
BLOCK_POLICIES = {1: "ar_b1", 8: "ao_b8", 16: "ao_b16", 32: "ao_b32"}
K64 = (1, 2, 4, 8, 16, 32, 64)
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPLICATES = 10_000


def pass_at_k(n: int, successes: int, k: int) -> float:
    if not 0 <= successes <= n or not 1 <= k <= n:
        raise ValueError("invalid pass@k inputs")
    if n - successes < k:
        return 1.0
    return 1.0 - math.comb(n - successes, k) / math.comb(n, k)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pooled_summary(values: dict[str, list[float]], label: str) -> dict[str, float]:
    point = statistics.mean(statistics.mean(values[task]) for task in TASKS)
    seed_material = f"{BOOTSTRAP_SEED}|postreview|{label}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in TASKS:
        source = np.asarray(values[task], dtype=float)
        indices = rng.integers(0, len(source), size=(BOOTSTRAP_REPLICATES, len(source)))
        task_draws.append(source[indices].mean(axis=1))
    draws = np.mean(np.stack(task_draws, axis=0), axis=0)
    return {
        "point": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def majority_vote_correct(records: list[dict]) -> float:
    """Expected correctness under uniform tie-breaking among modal parsed answers."""
    groups: dict[str, list[bool]] = {}
    for record in records:
        answer = record.get("parsed_answer")
        if answer is None or str(answer).strip() == "":
            continue
        groups.setdefault(str(answer), []).append(bool(record["correct"]))
    if not groups:
        return 0.0
    maximum = max(len(flags) for flags in groups.values())
    tied = [flags for flags in groups.values() if len(flags) == maximum]
    return statistics.mean(statistics.mean(flags) for flags in tied)


def phase1_path(root: Path, task: str, policy: str, row: int, rollout: int) -> Path:
    if rollout < 16:
        run_kind = "screening"
    elif policy in {"ao_b8", "ao_b16"}:
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


def record_path(root: Path, task: str, policy: str, row: int, rollout: int) -> Path:
    if policy == "ef_b32":
        return (
            root
            / "outputs/dllm_order_condition_a/condition-a-v1/condition-a-formal"
            / task
            / policy
            / f"q{row:04d}"
            / f"r{rollout:02d}.json"
        )
    if policy == "random_b32" and rollout >= 16:
        return (
            root
            / "outputs/dllm_order_phase1/phase1-v1/heldout/random-extension"
            / task
            / policy
            / f"q{row:04d}"
            / f"r{rollout:02d}.json"
        )
    return phase1_path(root, task, policy, row, rollout)


def load_queries(root: Path) -> dict[str, list[int]]:
    manifest = json.loads((root / "protocols/phase1_split_manifest_v1.json").read_text())
    return {
        task: sorted(int(row["row_index"]) for row in manifest["splits"][task]["heldout"])
        for task in TASKS
    }


def load_query_metadata(root: Path) -> dict[str, dict[int, dict]]:
    manifest = json.loads((root / "protocols/phase1_split_manifest_v1.json").read_text())
    return {
        task: {int(row["row_index"]): row for row in manifest["splits"][task]["heldout"]}
        for task in TASKS
    }


def load_all(root: Path, queries: dict[str, list[int]]) -> dict[tuple[str, int, str], list[dict]]:
    loaded = {}
    for task in TASKS:
        for row in queries[task]:
            for policy in POLICIES:
                records = []
                for rollout in range(64):
                    path = record_path(root, task, policy, row, rollout)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    records.append(json.loads(path.read_text()))
                loaded[(task, row, policy)] = records
    return loaded


def pass_contrast(
    records: dict,
    queries: dict[str, list[int]],
    left: str,
    right: str,
    indices: range,
    k: int,
    label: str,
) -> dict:
    n = len(indices)
    values = {task: [] for task in TASKS}
    for task in TASKS:
        for row in queries[task]:
            left_success = sum(records[(task, row, left)][i]["correct"] for i in indices)
            right_success = sum(records[(task, row, right)][i]["correct"] for i in indices)
            values[task].append(
                pass_at_k(n, left_success, k) - pass_at_k(n, right_success, k)
            )
    return pooled_summary(values, label)


def random_extension_checks(records: dict, queries: dict[str, list[int]]) -> dict:
    new_only = {
        str(k): pass_contrast(
            records, queries, "random_b32", "ao_b32", range(16, 64), k, f"new48-k{k}"
        )
        for k in (1, 2, 4, 8, 16, 32)
    }
    values = {task: [] for task in TASKS}
    for task in TASKS:
        for row in queries[task]:
            random_records = records[(task, row, "random_b32")]
            rnd = statistics.mean(
                pass_at_k(48, sum(r["correct"] for r in random_records[16:]), k)
                - pass_at_k(48, sum(r["correct"] for r in records[(task, row, "ao_b32")][16:]), k)
                for k in (16, 32)
            )
            values[task].append(rnd)

    old_new = {}
    for k in (1, 2, 4, 8, 16):
        contrasts = {task: [] for task in TASKS}
        for task in TASKS:
            for row in queries[task]:
                random_records = records[(task, row, "random_b32")]
                old = sum(r["correct"] for r in random_records[:16])
                new = sum(r["correct"] for r in random_records[16:])
                contrasts[task].append(pass_at_k(16, old, k) - pass_at_k(48, new, k))
        old_new[str(k)] = pooled_summary(contrasts, f"random-old16-minus-new48-k{k}")

    split_half = {}
    for name, indices in (("rollouts_0_31", range(0, 32)), ("rollouts_32_63", range(32, 64))):
        split_half[name] = {
            str(k): pass_contrast(
                records, queries, "random_b32", "ao_b32", indices, k, f"{name}-k{k}"
            )
            for k in (1, 2, 4, 8, 16, 32)
        }
    return {
        "new48_random_minus_cf": new_only,
        "new48_high_budget_mean_k16_32": pooled_summary(values, "new48-high-mean"),
        "random_old16_minus_new48": old_new,
        "split_half_random_minus_cf": split_half,
    }


def block_size_checks(records: dict, queries: dict[str, list[int]]) -> dict:
    per_k = {}
    for k in K64:
        per_k[str(k)] = {}
        for left, right in ((8, 1), (16, 8), (32, 16), (32, 1)):
            per_k[str(k)][f"b{left}_minus_b{right}"] = pass_contrast(
                records,
                queries,
                BLOCK_POLICIES[left],
                BLOCK_POLICIES[right],
                range(64),
                k,
                f"block-b{left}-b{right}-k{k}",
            )
    grids = {
        "low_k1_2_4_8": (1, 2, 4, 8),
        "frozen_k4_8_16_32_64": (4, 8, 16, 32, 64),
        "high_k16_32_64": (16, 32, 64),
    }
    grid_results = {}
    for name, grid in grids.items():
        values = {task: [] for task in TASKS}
        for task in TASKS:
            for row in queries[task]:
                b32 = sum(r["correct"] for r in records[(task, row, "ao_b32")])
                b1 = sum(r["correct"] for r in records[(task, row, "ar_b1")])
                values[task].append(
                    statistics.mean(pass_at_k(64, b32, k) - pass_at_k(64, b1, k) for k in grid)
                )
        grid_results[name] = pooled_summary(values, f"block-grid-{name}")
    return {"per_k": per_k, "b32_minus_b1_grid_sensitivity": grid_results}


def majority_vote_checks(records: dict, queries: dict[str, list[int]]) -> dict:
    result = {policy: {} for policy in POLICIES}
    for policy in POLICIES:
        for k in K64:
            values = {task: [] for task in TASKS}
            for task in TASKS:
                for row in queries[task]:
                    values[task].append(majority_vote_correct(records[(task, row, policy)][:k]))
            result[policy][str(k)] = {
                "pooled": statistics.mean(statistics.mean(values[t]) for t in TASKS),
                "per_task": {task: statistics.mean(values[task]) for task in TASKS},
            }
    return result


def quality_checks(records: dict, queries: dict[str, list[int]]) -> dict:
    result = {task: {} for task in TASKS}
    for task in TASKS:
        for policy in POLICIES:
            cells = [records[(task, row, policy)] for row in queries[task]]
            flat = [record for cell in cells for record in cell]
            counts = sorted(sum(record["correct"] for record in cell) for cell in cells)
            result[task][policy] = {
                "parser_status_counts": dict(sorted(Counter(r["parser_status"] for r in flat).items())),
                "empty_parsed_answers": sum(
                    r.get("parsed_answer") is None or str(r.get("parsed_answer")).strip() == ""
                    for r in flat
                ),
                "empty_responses": sum(not str(r.get("response", "")).strip() for r in flat),
                "mean_whitespace_tokens": statistics.mean(len(str(r["response"]).split()) for r in flat),
                "zero_success_queries": sum(count == 0 for count in counts),
                "success_count_quantiles": {
                    "min": counts[0],
                    "q25": percentile(counts, 0.25),
                    "median": percentile(counts, 0.5),
                    "q75": percentile(counts, 0.75),
                    "max": counts[-1],
                },
                "complete_256_event_traces": sum(
                    r.get("completion_status") == "complete" and r.get("trace_event_count") == 256
                    for r in flat
                ),
                "records": len(flat),
            }
    return result


def math_difficulty_checks(records: dict, metadata: dict[str, dict[int, dict]]) -> dict:
    """Descriptive MATH-500 breakdown using the frozen manifest strata."""
    rows_by_level: dict[str, list[int]] = {}
    for row, row_metadata in metadata["math500"].items():
        level = str(row_metadata["stratum"]).rsplit("|", 1)[-1]
        rows_by_level.setdefault(level, []).append(row)

    result = {}
    for level in sorted(rows_by_level, key=lambda value: int(value.split()[-1])):
        rows = sorted(rows_by_level[level])
        policy_results = {}
        for policy in POLICIES:
            pass1 = []
            coverage = []
            for row in rows:
                successes = sum(r["correct"] for r in records[("math500", row, policy)])
                pass1.append(successes / 64)
                coverage.append(
                    statistics.mean(pass_at_k(64, successes, k) for k in (4, 8, 16, 32, 64))
                )
            policy_results[policy] = {
                "pass_at_1": statistics.mean(pass1),
                "coverage_auc_k4_8_16_32_64": statistics.mean(coverage),
            }
        result[level] = {"queries": len(rows), "policies": policy_results}
    return {
        "status": "post_outcome_descriptive_no_multiplicity_adjustment",
        "levels": result,
    }


def generation_accounting() -> dict:
    rows = {
        "phase0_signal": 3_200,
        "phase1_calibration": 6_400,
        "phase1_heldout_screening": 16_000,
        "phase1_primary_extension": 19_200,
        "phase2_block_size_extension": 19_200,
        "condition_a_entropy_first": 12_800,
        "random_policy_extension": 9_600,
    }
    return {"rows": rows, "total": sum(rows.values()), "excludes": "smoke and replay checks"}


def interval(value: dict) -> str:
    return f"{value['point']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]"


def render(report: dict) -> str:
    random_checks = report["random_extension"]
    lines = [
        "# Post-Review CPU Audits",
        "",
        "**Status:** post-outcome exploratory analyses. No frozen endpoint, interval,",
        "or verdict is replaced by this report.",
        "",
        "## Random-extension independence checks",
        "",
        "| k | New-48 RND - CF | Old-16 minus new-48 RND |",
        "|---:|---:|---:|",
    ]
    for k in (1, 2, 4, 8, 16):
        lines.append(
            f"| {k} | {interval(random_checks['new48_random_minus_cf'][str(k)])} | "
            f"{interval(random_checks['random_old16_minus_new48'][str(k)])} |"
        )
    lines.extend(
        [
            f"| 32 | {interval(random_checks['new48_random_minus_cf']['32'])} | n/a |",
            "",
            "New-48 mean over k={16,32}: "
            + interval(random_checks["new48_high_budget_mean_k16_32"]),
            "",
            "## Block-size grid sensitivity",
            "",
            "| Grid | B32 - B1 |",
            "|---|---:|",
        ]
    )
    for name, summary in report["block_size"]["b32_minus_b1_grid_sensitivity"].items():
        lines.append(f"| `{name}` | {interval(summary)} |")
    lines.extend(
        [
            "",
            "## MATH-500 difficulty stratification",
            "",
            "Descriptive only; strata come from the frozen split manifest.",
            "",
            "| Level | n | Sequential CAUC | CF CAUC | RND CAUC | EF CAUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for level, level_result in report["math_difficulty"]["levels"].items():
        policies = level_result["policies"]
        lines.append(
            f"| {level} | {level_result['queries']} | "
            f"{policies['ar_b1']['coverage_auc_k4_8_16_32_64']:.3f} | "
            f"{policies['ao_b32']['coverage_auc_k4_8_16_32_64']:.3f} | "
            f"{policies['random_b32']['coverage_auc_k4_8_16_32_64']:.3f} | "
            f"{policies['ef_b32']['coverage_auc_k4_8_16_32_64']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Generation accounting",
            "",
            "| Stage | New generations |",
            "|---|---:|",
        ]
    )
    for name, count in report["generation_accounting"]["rows"].items():
        lines.append(f"| `{name}` | {count:,} |")
    lines.extend(
        [
            f"| **Total** | **{report['generation_accounting']['total']:,}** |",
            "",
            "The JSON additionally contains per-k block contrasts, split-half checks,",
            "tie-aware majority-vote accuracy, parser/empty-response summaries, decoded",
            "lengths, and per-query success-count distributions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    queries = load_queries(args.root)
    metadata = load_query_metadata(args.root)
    records = load_all(args.root, queries)
    report = {
        "analysis_status": "post_outcome_exploratory_cpu_only",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_extension": random_extension_checks(records, queries),
        "block_size": block_size_checks(records, queries),
        "majority_vote": majority_vote_checks(records, queries),
        "quality": quality_checks(records, queries),
        "math_difficulty": math_difficulty_checks(records, metadata),
        "generation_accounting": generation_accounting(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "postreview_cpu_audits.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "postreview_cpu_audits.md").write_text(render(report))


if __name__ == "__main__":
    main()
