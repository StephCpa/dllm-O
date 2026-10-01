"""Freeze, validate, and analyze the post-Gate random-policy extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .artifacts import sha256_file, write_json_atomic, write_text_atomic
from .metrics import estimate_pass_at_k
from .phase1_protocol import (
    BCI_FREEZE_SHA256,
    JUSTGRPO_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_SEED,
    PROTOCOL_VERSION,
    RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256,
    SPLIT_MANIFEST_SHA256,
    build_phase1_work_items,
    load_split_manifest,
    validate_random_extension_freeze,
)
from .seeding import rollout_seed_key


FREEZE_VERSION = "random-extension-freeze-v1"
TASKS = ("gsm8k", "math500")
POLICIES = ("random_b32", "ao_b32", "ar_b1", "ef_b32")
NEW_ROLLOUTS = tuple(range(16, 64))
ALL_ROLLOUTS = tuple(range(64))
PRIMARY_GRID = (16, 32, 64)
LOW_GRID = (2, 4, 8, 16)
FULL_GRID = (4, 8, 16, 32, 64)
REPORT_GRID = (1, 2, 4, 8, 16, 32, 64)
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_REPLICATES = 10_000
DEFAULT_NUM_SHARDS = 3
EXPECTED_NEW_GENERATIONS = 9_600


class RandomExtensionError(RuntimeError):
    pass


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _freeze_payload(runner_commit: str) -> dict[str, Any]:
    return {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "post-unblinding targeted prospective follow-up",
        "selection_rationale": (
            "A post-unblinding matched first-16 decomposition found AO-random "
            "contrasts positive at low k and directionally negative at k=8/16; "
            "random_b32 was the only frontier policy without k=32/64 data."
        ),
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "bci_freeze_sha256": BCI_FREEZE_SHA256,
        "artifact_schema_sha256": RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256,
        "runner_commit": runner_commit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "external_repository_commit": JUSTGRPO_COMMIT,
        "policy": "random_b32",
        "rollout_budget": {
            "existing_indices": [0, 15],
            "new_indices": [16, 63],
            "analysis_rollouts_per_query": 64,
        },
        "primary_endpoint": {
            "name": "HighBudgetCoverage64",
            "definition": "mean unbiased Pass@k over k in {16,32,64}",
            "contrast": "random_b32 - ao_b32",
            "unit": "paired held-out query",
            "aggregation": "equal-weight mean of GSM8K and MATH-500 task means",
            "uncertainty": {
                "method": "task-stratified query bootstrap percentile interval",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "level": 0.95,
            },
        },
        "secondary_endpoints": [
            "per-task primary contrast",
            "random minus AO/AR/EF Pass@k for k in {1,2,4,8,16,32,64}",
            "low grid {2,4,8,16} and full grid {4,8,16,32,64}",
            "k=64 solved-set discordance",
            "parsed-answer diversity at k=16 and k=64",
            "parser, empty-response, decoded-length, and completeness summaries",
        ],
        "verdict_taxonomy": {
            "ci_entirely_above_zero": "resolved random high-budget coverage benefit",
            "ci_straddles_zero": "no resolved high-budget difference",
            "ci_entirely_below_zero": "resolved random high-budget coverage harm",
        },
        "cardinality": {
            "tasks": list(TASKS),
            "heldout_queries_per_task": 100,
            "new_rollouts_per_query": 48,
            "new_generations": EXPECTED_NEW_GENERATIONS,
            "num_shards": DEFAULT_NUM_SHARDS,
            "generations_per_shard": EXPECTED_NEW_GENERATIONS
            // DEFAULT_NUM_SHARDS,
        },
        "analysis_rules": {
            "parser_failures_and_empty_responses": "retain as incorrect",
            "missing_or_invalid_artifact": "block analysis",
            "secondary_endpoints_change_primary_verdict": False,
            "query_exclusions_after_outcome_read": False,
            "first_correctness_read_must_be_logged": True,
        },
        "not_permitted": [
            "calling the extension part of the original Gate-1 chain",
            "reinterpreting the frozen AO-random aggregate null",
            "universal random-ranking superiority or crossover claims",
            "treating answer-string diversity as semantic reasoning diversity",
            "adding an endpoint, policy, task, or exclusion after outcome read",
        ],
    }


def emit_freeze(output: Path, repo_root: Path) -> str:
    if output.exists():
        raise RandomExtensionError(f"refusing to overwrite freeze: {output}")
    payload = _freeze_payload(_repo_commit(repo_root))
    write_json_atomic(output, payload)
    validate_random_extension_freeze(output, SPLIT_MANIFEST_SHA256)
    return sha256_file(output)


def _schema(path: Path) -> Draft202012Validator:
    if sha256_file(path) != RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256:
        raise RandomExtensionError("random-extension artifact schema hash mismatch")
    payload = json.loads(path.read_text())
    Draft202012Validator.check_schema(payload)
    return Draft202012Validator(payload)


def _queries(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        task: sorted(
            manifest["splits"][task]["heldout"], key=lambda row: row["row_index"]
        )
        for task in TASKS
    }


def _new_path(root: Path, task: str, row: int, rollout: int) -> Path:
    return (
        root
        / PROTOCOL_VERSION
        / "heldout"
        / "random-extension"
        / task
        / "random_b32"
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def _trace_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.stem + ".trace.jsonl.gz")


def _marker_path(root: Path, shard: int) -> Path:
    return (
        root
        / PROTOCOL_VERSION
        / "manifests"
        / "heldout"
        / "random-extension"
        / "all"
        / f"shard-{shard:03d}-of-{DEFAULT_NUM_SHARDS:03d}.json"
    )


def _hash_keys(keys: list[str]) -> str:
    payload = "\n".join(sorted(keys)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_artifacts(
    *,
    phase1_root: Path,
    manifest: dict[str, Any],
    freeze_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    freeze_sha = validate_random_extension_freeze(
        freeze_path, SPLIT_MANIFEST_SHA256
    )
    freeze = json.loads(freeze_path.read_text())
    validator = _schema(schema_path)
    present = 0
    missing = []
    invalid = []
    for task, entries in _queries(manifest).items():
        for entry in entries:
            row = int(entry["row_index"])
            for rollout in NEW_ROLLOUTS:
                path = _new_path(phase1_root, task, row, rollout)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                try:
                    record = json.loads(path.read_text())
                    validator.validate(record)
                    expected = {
                        "protocol_version": PROTOCOL_VERSION,
                        "run_kind": "random-extension",
                        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                        "bci_freeze_sha256": BCI_FREEZE_SHA256,
                        "random_extension_freeze_sha256": freeze_sha,
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "rollout_index": rollout,
                        "seed_key": rollout_seed_key(
                            row,
                            rollout,
                            global_seed=PROTOCOL_SEED,
                            protocol_prefix=f"dllm-order-phase1-v1|{task}",
                        ),
                        "external_repository_commit": JUSTGRPO_COMMIT,
                        "completion_status": "complete",
                    }
                    if any(record.get(key) != value for key, value in expected.items()):
                        raise RandomExtensionError("record provenance mismatch")
                    if record["runner_repository"] != {
                        "commit": freeze["runner_commit"],
                        "dirty": False,
                    }:
                        raise RandomExtensionError("runner commit mismatch")
                    if record["policy"]["name"] != "random_b32":
                        raise RandomExtensionError("policy mismatch")
                    source = record["source"]
                    expected_source = manifest["sources"][task]
                    if (
                        source["row_index"] != row
                        or source["split_role"] != "heldout"
                        or source["prompt_sha256"] != entry["prompt_sha256"]
                        or source["dataset_id"] != expected_source["dataset_id"]
                        or source["dataset_revision"] != expected_source["revision"]
                        or source["dataset_split"] != expected_source["split"]
                    ):
                        raise RandomExtensionError("source mismatch")
                    trace = _trace_path(path)
                    if not trace.is_file() or sha256_file(trace) != record["trace_sha256"]:
                        raise RandomExtensionError("trace hash mismatch")
                except Exception as error:
                    invalid.append(f"{path}: {error}")
                    continue
                present += 1

    marker_count = 0
    marker_results = 0
    marker_errors = []
    for shard in range(DEFAULT_NUM_SHARDS):
        path = _marker_path(phase1_root, shard)
        if not path.is_file():
            marker_errors.append(f"missing {path}")
            continue
        try:
            marker = json.loads(path.read_text())
            validator.validate(marker)
            expected_items = build_phase1_work_items(
                manifest,
                split_role="heldout",
                run_kind="random-extension",
                task_selection="all",
                shard_index=shard,
                num_shards=DEFAULT_NUM_SHARDS,
            )
            expected_key_hash = _hash_keys([item.key for item in expected_items])
            expected_fields = {
                "protocol_version": PROTOCOL_VERSION,
                "split_role": "heldout",
                "run_kind": "random-extension",
                "task_selection": "all",
                "shard_index": shard,
                "num_shards": DEFAULT_NUM_SHARDS,
                "result_count": len(expected_items),
                "expected_keys_sha256": expected_key_hash,
                "observed_keys_sha256": expected_key_hash,
            }
            if any(marker.get(key) != value for key, value in expected_fields.items()):
                raise RandomExtensionError("marker provenance or key set mismatch")
            if marker["result_count"] != EXPECTED_NEW_GENERATIONS // DEFAULT_NUM_SHARDS:
                raise RandomExtensionError("marker result count mismatch")
        except Exception as error:
            marker_errors.append(f"{path}: {error}")
            continue
        marker_count += 1
        marker_results += marker["result_count"]
    return {
        "expected": EXPECTED_NEW_GENERATIONS,
        "present_and_valid": present,
        "missing": len(missing),
        "invalid": len(invalid),
        "missing_examples": missing[:10],
        "invalid_examples": invalid[:10],
        "markers": marker_count,
        "marker_results": marker_results,
        "marker_errors": marker_errors,
        "outcome_fields_read": False,
    }


def _old_path(
    root: Path, task: str, policy: str, row: int, rollout: int
) -> Path:
    if policy == "ef_b32":
        raise ValueError("EF records use the Condition A root")
    run_kind = "screening" if rollout < 16 else "primary-extension"
    return (
        root
        / PROTOCOL_VERSION
        / "heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def _ef_path(root: Path, task: str, row: int, rollout: int) -> Path:
    return (
        root
        / "condition-a-v1"
        / "condition-a-formal"
        / task
        / "ef_b32"
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def _load_policy_records(
    *,
    phase1_root: Path,
    condition_root: Path,
    task: str,
    row: int,
    policy: str,
) -> list[dict[str, Any]]:
    paths = []
    for rollout in ALL_ROLLOUTS:
        if policy == "random_b32":
            path = (
                _old_path(phase1_root, task, policy, row, rollout)
                if rollout < 16
                else _new_path(phase1_root, task, row, rollout)
            )
        elif policy == "ef_b32":
            path = _ef_path(condition_root, task, row, rollout)
        else:
            path = _old_path(phase1_root, task, policy, row, rollout)
        if not path.is_file():
            raise RandomExtensionError(f"missing analysis input: {path}")
        paths.append(path)
    return [json.loads(path.read_text()) for path in paths]


def _coverage(successes: int, grid: tuple[int, ...]) -> float:
    return statistics.mean(estimate_pass_at_k(64, successes, k) for k in grid)


def _mean_tasks(values: dict[str, list[float]]) -> float:
    return statistics.mean(statistics.mean(rows) for rows in values.values())


def _quantile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _bootstrap(
    values: dict[str, list[float]], label: str
) -> dict[str, float]:
    rng = random.Random(f"{BOOTSTRAP_SEED}|random-extension|{label}")
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = {
            task: [
                values[task][rng.randrange(len(values[task]))]
                for _ in values[task]
            ]
            for task in values
        }
        draws.append(_mean_tasks(sampled))
    draws.sort()
    return {
        "point": _mean_tasks(values),
        "ci_low": _quantile(draws, 0.025),
        "ci_high": _quantile(draws, 0.975),
    }


def _classify(interval: dict[str, float]) -> str:
    if interval["ci_low"] > 0:
        return "entirely_above_zero"
    if interval["ci_high"] < 0:
        return "entirely_below_zero"
    return "straddles_zero"


def _answer_entropy(answers: list[str | None]) -> float:
    counts = Counter(answer for answer in answers if answer is not None)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def analyze(
    *,
    phase1_root: Path,
    condition_root: Path,
    manifest: dict[str, Any],
    freeze_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    validation = validate_artifacts(
        phase1_root=phase1_root,
        manifest=manifest,
        freeze_path=freeze_path,
        schema_path=schema_path,
    )
    if (
        validation["present_and_valid"] != EXPECTED_NEW_GENERATIONS
        or validation["missing"]
        or validation["invalid"]
        or validation["markers"] != DEFAULT_NUM_SHARDS
        or validation["marker_results"] != EXPECTED_NEW_GENERATIONS
        or validation["marker_errors"]
    ):
        raise RandomExtensionError("artifact closure failed; refusing outcome read")

    records: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    counts: dict[tuple[str, int, str], int] = {}
    for task, entries in _queries(manifest).items():
        for entry in entries:
            row = int(entry["row_index"])
            for policy in POLICIES:
                loaded = _load_policy_records(
                    phase1_root=phase1_root,
                    condition_root=condition_root,
                    task=task,
                    row=row,
                    policy=policy,
                )
                records[(task, row, policy)] = loaded
                counts[(task, row, policy)] = sum(
                    bool(record["correct"]) for record in loaded
                )

    primary_values = {task: [] for task in TASKS}
    for task, entries in _queries(manifest).items():
        for entry in entries:
            row = int(entry["row_index"])
            primary_values[task].append(
                _coverage(counts[(task, row, "random_b32")], PRIMARY_GRID)
                - _coverage(counts[(task, row, "ao_b32")], PRIMARY_GRID)
            )
    primary = _bootstrap(primary_values, "primary")

    pairwise = {}
    for baseline in ("ao_b32", "ar_b1", "ef_b32"):
        per_k = {}
        for k in REPORT_GRID:
            values = {task: [] for task in TASKS}
            for task, entries in _queries(manifest).items():
                for entry in entries:
                    row = int(entry["row_index"])
                    values[task].append(
                        estimate_pass_at_k(
                            64, counts[(task, row, "random_b32")], k
                        )
                        - estimate_pass_at_k(64, counts[(task, row, baseline)], k)
                    )
            per_k[str(k)] = {
                "pooled": _bootstrap(values, f"{baseline}|k{k}"),
                "per_task_points": {
                    task: statistics.mean(values[task]) for task in TASKS
                },
            }
        grids = {}
        for name, grid in (
            ("low_k2_4_8_16", LOW_GRID),
            ("full_k4_8_16_32_64", FULL_GRID),
        ):
            values = {task: [] for task in TASKS}
            for task, entries in _queries(manifest).items():
                for entry in entries:
                    row = int(entry["row_index"])
                    values[task].append(
                        _coverage(counts[(task, row, "random_b32")], grid)
                        - _coverage(counts[(task, row, baseline)], grid)
                    )
            grids[name] = _bootstrap(values, f"{baseline}|{name}")
        pairwise[f"random_minus_{baseline}"] = {"per_k": per_k, "grids": grids}

    summaries = {task: {} for task in TASKS}
    solved = {task: {} for task in TASKS}
    quality = {task: {} for task in TASKS}
    for task, entries in _queries(manifest).items():
        for policy in POLICIES:
            success_counts = [
                counts[(task, int(entry["row_index"]), policy)] for entry in entries
            ]
            summaries[task][policy] = {
                f"pass_at_{k}": statistics.mean(
                    estimate_pass_at_k(64, count, k) for count in success_counts
                )
                for k in REPORT_GRID
            }
            policy_records = [
                record
                for entry in entries
                for record in records[(task, int(entry["row_index"]), policy)]
            ]
            quality[task][policy] = {
                "parser_status_counts": dict(
                    sorted(Counter(record["parser_status"] for record in policy_records).items())
                ),
                "empty_response_rate": statistics.mean(
                    not str(record["response"]).strip() for record in policy_records
                ),
                "mean_whitespace_tokens": statistics.mean(
                    len(str(record["response"]).split()) for record in policy_records
                ),
            }
            summaries[task][policy]["diversity"] = {}
            for k in (16, 64):
                unique = []
                entropy = []
                modal = []
                for entry in entries:
                    row = int(entry["row_index"])
                    answers = [
                        record["parsed_answer"]
                        for record in records[(task, row, policy)][:k]
                    ]
                    answer_counts = Counter(
                        answer for answer in answers if answer is not None
                    )
                    unique.append(len(answer_counts))
                    entropy.append(_answer_entropy(answers))
                    modal.append(max(answer_counts.values()) / k if answer_counts else 0.0)
                summaries[task][policy]["diversity"][f"k{k}"] = {
                    "mean_unique_answers": statistics.mean(unique),
                    "mean_answer_entropy": statistics.mean(entropy),
                    "mean_modal_share": statistics.mean(modal),
                }

        for baseline in ("ao_b32", "ar_b1", "ef_b32"):
            membership = {"random_only": 0, "baseline_only": 0, "both": 0, "neither": 0}
            for entry in entries:
                row = int(entry["row_index"])
                random_solved = counts[(task, row, "random_b32")] > 0
                baseline_solved = counts[(task, row, baseline)] > 0
                if random_solved and baseline_solved:
                    membership["both"] += 1
                elif random_solved:
                    membership["random_only"] += 1
                elif baseline_solved:
                    membership["baseline_only"] += 1
                else:
                    membership["neither"] += 1
            solved[task][f"random_vs_{baseline}"] = membership

    return {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha256_file(freeze_path),
        "validation": validation,
        "primary": {
            "contrast": "HighBudgetCoverage(random_b32)-HighBudgetCoverage(ao_b32)",
            "grid": list(PRIMARY_GRID),
            "pooled": primary,
            "classification": _classify(primary),
            "per_task": {
                task: _bootstrap(
                    {task: primary_values[task]}, f"primary|{task}"
                )
                for task in TASKS
            },
        },
        "secondary_pairwise": pairwise,
        "summaries": summaries,
        "solved_set_membership": solved,
        "quality": quality,
    }


def _interval(value: dict[str, float]) -> str:
    return (
        f"{value['point']:+.5f} "
        f"[{value['ci_low']:+.5f}, {value['ci_high']:+.5f}]"
    )


def render_report(report: dict[str, Any]) -> str:
    primary = report["primary"]
    lines = [
        "# Random-B32 64-Rollout Extension Results",
        "",
        "**Evidence status:** post-unblinding-selected targeted prospective follow-up;",
        "not part of the original Gate-1 confirmatory chain.",
        "",
        "## Artifact closure",
        "",
        f"- Valid new records: {report['validation']['present_and_valid']}/9600.",
        f"- Shard markers: {report['validation']['markers']}/3.",
        "",
        "## Frozen primary endpoint",
        "",
        f"- Grid: `{primary['grid']}`.",
        f"- Random minus AO: {_interval(primary['pooled'])}.",
        f"- Classification: `{primary['classification']}`.",
        "",
        "## Per-k secondary contrasts",
        "",
        "| Baseline | k | Random - baseline, pooled 95% CI |",
        "|---|---:|---:|",
    ]
    for key, comparison in report["secondary_pairwise"].items():
        baseline = key.removeprefix("random_minus_")
        for k, result in comparison["per_k"].items():
            lines.append(f"| `{baseline}` | {k} | {_interval(result['pooled'])} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The primary taxonomy is frozen in the machine-readable protocol. Secondary",
            "comparisons characterize the frontier and cannot modify the primary verdict.",
            "This extension was selected after seeing the first-16 crossing and cannot be",
            "presented as part of the original pre-registered sequence.",
            "",
        ]
    )
    return "\n".join(lines)


def _append_invocation(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["emit-freeze", "validate-only", "analyze"],
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--condition-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--invocation-log", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "emit-freeze":
        freeze_sha = emit_freeze(args.freeze, args.repo_root)
        print(json.dumps({"freeze": str(args.freeze), "sha256": freeze_sha}, indent=2))
        return 0

    if args.phase1_root is None:
        raise RandomExtensionError("validation and analysis require --phase1-root")
    manifest = load_split_manifest(args.split_manifest)
    event = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "reads_random_extension_correctness": args.mode == "analyze",
        "freeze_sha256": sha256_file(args.freeze),
        "analyzer_commit": _repo_commit(args.repo_root),
    }
    if args.invocation_log is not None:
        _append_invocation(args.invocation_log, event)

    if args.mode == "validate-only":
        result = validate_artifacts(
            phase1_root=args.phase1_root,
            manifest=manifest,
            freeze_path=args.freeze,
            schema_path=args.artifact_schema,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.condition_root is None or args.output_json is None or args.output_md is None:
        raise RandomExtensionError(
            "analysis requires --condition-root, --output-json, and --output-md"
        )
    report = analyze(
        phase1_root=args.phase1_root,
        condition_root=args.condition_root,
        manifest=manifest,
        freeze_path=args.freeze,
        schema_path=args.artifact_schema,
    )
    write_json_atomic(args.output_json, report)
    write_text_atomic(args.output_md, render_report(report))
    print(json.dumps(report["primary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
