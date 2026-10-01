"""Inventory, freeze, validate, and analyze the temperature extension."""

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
    SPLIT_MANIFEST_SHA256,
    load_split_manifest,
)
from .seeding import rollout_seed_key
from .temperature_extension_protocol import (
    ARTIFACT_SCHEMA_SHA256,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFAULT_NUM_SHARDS,
    EXPECTED_FORMAL_GENERATIONS,
    FORMAL_ROLLOUTS,
    FREEZE_VERSION,
    HIGH_BUDGET_GRID,
    K_GRID,
    POLICY_BY_NAME,
    PROTOCOL_VERSION,
    TEMPERATURES,
    TEMPERATURE_POLICIES,
    build_work_items,
    validate_freeze,
)


TASKS = ("gsm8k", "math500")
CONTROL_INVENTORY_VERSION = "temperature-control-inventory-v1"
CONTROL_ROLLOUTS = tuple(range(32))
CONTROL_POLICY_BY_SCHEDULE = {
    "sequential": "ar_b1",
    "confidence_first": "ao_b32",
    "random": "random_b32",
}
EXPECTED_CONTROL_ARTIFACTS = 19_200
RANDOM_EXTENSION_FREEZE_SHA256 = (
    "5b84f29e1b57415c654bdf377dd775fa1bbe098679215a51c2e46a06e36d44d3"
)


class TemperatureExtensionError(RuntimeError):
    pass


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repo(repo_root: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise TemperatureExtensionError(
            "refusing to freeze from a dirty repository; commit protocol code first"
        )


def _queries(
    manifest: dict[str, Any], split_role: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        task: sorted(
            manifest["splits"][task][split_role], key=lambda row: row["row_index"]
        )
        for task in TASKS
    }


def _control_result_path(
    root: Path, task: str, row: int, schedule: str, rollout: int
) -> Path:
    policy = CONTROL_POLICY_BY_SCHEDULE[schedule]
    if rollout < 16:
        run_kind = "screening"
    elif schedule == "random":
        run_kind = "random-extension"
    else:
        run_kind = "primary-extension"
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


def _new_result_path(
    root: Path, run_kind: str, task: str, policy: str, row: int, rollout: int
) -> Path:
    return (
        root
        / PROTOCOL_VERSION
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def _trace_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.stem + ".trace.jsonl.gz")


def _seed_key(task: str, row: int, rollout: int) -> str:
    return rollout_seed_key(
        row,
        rollout,
        global_seed=PROTOCOL_SEED,
        protocol_prefix=f"dllm-order-phase1-v1|{task}",
    )


def _expected_control_policy(schedule: str) -> dict[str, Any]:
    return {
        "name": CONTROL_POLICY_BY_SCHEDULE[schedule],
        "steps": 256,
        "generation_length": 256,
        "block_length": 1 if schedule == "sequential" else 32,
        "temperature": 0.6,
        "cfg_scale": 0.0,
        "remasking": "random" if schedule == "random" else "low_confidence",
    }


def _validate_control_record(
    *,
    record: dict[str, Any],
    manifest: dict[str, Any],
    task: str,
    row: int,
    schedule: str,
    rollout: int,
) -> None:
    entry = next(
        item
        for item in manifest["splits"][task]["heldout"]
        if int(item["row_index"]) == row
    )
    source = manifest["sources"][task]
    expected_run_kind = (
        "screening"
        if rollout < 16
        else ("random-extension" if schedule == "random" else "primary-extension")
    )
    checks = (
        record.get("protocol_version") == "phase1-v1",
        record.get("run_kind") == expected_run_kind,
        record.get("split_manifest_sha256") == SPLIT_MANIFEST_SHA256,
        record.get("bci_freeze_sha256") == BCI_FREEZE_SHA256,
        record.get("model_id") == MODEL_ID,
        record.get("model_revision") == MODEL_REVISION,
        record.get("external_repository_commit") == JUSTGRPO_COMMIT,
        record.get("policy") == _expected_control_policy(schedule),
        record.get("rollout_index") == rollout,
        record.get("seed_key") == _seed_key(task, row, rollout),
        record.get("completion_status") == "complete",
        record.get("runner_repository", {}).get("dirty") is False,
        record.get("source", {}).get("row_index") == row,
        record.get("source", {}).get("split_role") == "heldout",
        record.get("source", {}).get("prompt_sha256") == entry["prompt_sha256"],
        record.get("source", {}).get("dataset_id") == source["dataset_id"],
        record.get("source", {}).get("dataset_revision") == source["revision"],
        record.get("source", {}).get("dataset_split") == source["split"],
    )
    if not all(checks):
        raise TemperatureExtensionError(
            f"invalid T=0.6 control: {task}/{schedule}/q{row}/r{rollout}"
        )
    if (
        schedule == "random"
        and rollout >= 16
        and record.get("random_extension_freeze_sha256")
        != RANDOM_EXTENSION_FREEZE_SHA256
    ):
        raise TemperatureExtensionError(
            f"random-extension freeze mismatch: {task}/q{row}/r{rollout}"
        )


def build_control_inventory(
    *, phase1_root: Path, manifest: dict[str, Any], output: Path
) -> dict[str, Any]:
    if output.exists():
        raise TemperatureExtensionError(f"refusing to overwrite inventory: {output}")
    entries: list[dict[str, Any]] = []
    for task, queries in _queries(manifest, "heldout").items():
        for query in queries:
            row = int(query["row_index"])
            for schedule in CONTROL_POLICY_BY_SCHEDULE:
                for rollout in CONTROL_ROLLOUTS:
                    path = _control_result_path(
                        phase1_root, task, row, schedule, rollout
                    )
                    trace = _trace_path(path)
                    if not path.is_file() or not trace.is_file():
                        raise TemperatureExtensionError(
                            f"missing control artifact: {path}"
                        )
                    record = json.loads(path.read_text(encoding="utf-8"))
                    _validate_control_record(
                        record=record,
                        manifest=manifest,
                        task=task,
                        row=row,
                        schedule=schedule,
                        rollout=rollout,
                    )
                    trace_sha = sha256_file(trace)
                    if record.get("trace_sha256") != trace_sha:
                        raise TemperatureExtensionError(
                            f"control trace hash mismatch: {trace}"
                        )
                    entries.append(
                        {
                            "key": f"{task}|q{row:04d}|{schedule}|r{rollout:02d}",
                            "relative_result_path": str(path.relative_to(phase1_root)),
                            "result_sha256": sha256_file(path),
                            "trace_sha256": trace_sha,
                            "runner_commit": record["runner_repository"]["commit"],
                        }
                    )
    if len(entries) != EXPECTED_CONTROL_ARTIFACTS:
        raise TemperatureExtensionError("unexpected control inventory cardinality")
    if len({entry["key"] for entry in entries}) != EXPECTED_CONTROL_ARTIFACTS:
        raise TemperatureExtensionError("control inventory contains duplicate keys")
    payload = {
        "inventory_version": CONTROL_INVENTORY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "outcome_fields_read": False,
        "temperature": 0.6,
        "rollout_indices": [0, 31],
        "schedules": list(CONTROL_POLICY_BY_SCHEDULE),
        "artifact_count": len(entries),
        "entries": entries,
    }
    write_json_atomic(output, payload)
    return {"path": str(output), "sha256": sha256_file(output), "count": len(entries)}


def validate_control_inventory(
    *, phase1_root: Path, inventory_path: Path, expected_sha256: str
) -> dict[str, Any]:
    if sha256_file(inventory_path) != expected_sha256:
        raise TemperatureExtensionError("control inventory hash mismatch")
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        payload.get("inventory_version") != CONTROL_INVENTORY_VERSION
        or payload.get("split_manifest_sha256") != SPLIT_MANIFEST_SHA256
        or payload.get("outcome_fields_read") is not False
        or payload.get("artifact_count") != EXPECTED_CONTROL_ARTIFACTS
    ):
        raise TemperatureExtensionError("invalid control inventory header")
    for entry in payload["entries"]:
        path = phase1_root / entry["relative_result_path"]
        trace = _trace_path(path)
        if (
            not path.is_file()
            or not trace.is_file()
            or sha256_file(path) != entry["result_sha256"]
            or sha256_file(trace) != entry["trace_sha256"]
        ):
            raise TemperatureExtensionError(f"control artifact changed: {path}")
    return {
        "expected": EXPECTED_CONTROL_ARTIFACTS,
        "present_and_hash_matched": EXPECTED_CONTROL_ARTIFACTS,
        "outcome_fields_read": False,
    }


def build_freeze_payload(
    *, runner_commit: str, control_inventory_sha256: str
) -> dict[str, Any]:
    return {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "post-review prospective follow-up",
        "selection_rationale": (
            "Reviewers identified sampling temperature as the strongest alternative "
            "explanation for schedule-dependent precision-coverage frontiers."
        ),
        "runner_commit": runner_commit,
        "analyzer_commit": runner_commit,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "artifact_schema_sha256": ARTIFACT_SCHEMA_SHA256,
        "reused_control_inventory_sha256": control_inventory_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "external_repository_commit": JUSTGRPO_COMMIT,
        "protocol_seed": PROTOCOL_SEED,
        "reused_temperature": 0.6,
        "new_temperatures": list(TEMPERATURES),
        "new_policy_names": [policy.name for policy in TEMPERATURE_POLICIES],
        "paired_seed_construction": (
            "same Phase-1 seed key for a task/query/rollout across cells; not claimed "
            "to preserve an identical random trajectory after decoding paths diverge"
        ),
        "primary_family": {
            "contrast": "random B32 minus confidence-first B32",
            "temperatures": list(TEMPERATURES),
            "endpoints_per_temperature": {
                "low": "Pass@1",
                "high": "mean unbiased Pass@k over k in {16,32}",
            },
            "unit": "paired held-out query",
            "aggregation": "equal-weight mean of GSM8K and MATH-500 task means",
            "uncertainty": {
                "method": (
                    "task-stratified paired query bootstrap with nonstudentized "
                    "maximum-absolute-deviation simultaneous 95% bands"
                ),
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "family_size": 4,
            },
            "verdict": (
                "a reversal replicates at a temperature only if the simultaneous "
                "CI for the low endpoint is entirely below zero and the simultaneous "
                "CI for the high endpoint is entirely above zero"
            ),
        },
        "secondary_endpoints": [
            "per-task primary-family point estimates",
            "Pass@k curves for all schedules and temperatures on k={1,2,4,8,16,32}",
            "within-temperature per-k schedule contrasts",
            "cross-temperature empirical frontier comparisons",
            (
                "parsed-answer diversity and majority-vote accuracy; ties among "
                "modal parsed answers receive fractional correctness"
            ),
            "parser, empty-response, decoded-length, and completeness summaries",
        ],
        "cardinality": {
            "tasks": list(TASKS),
            "heldout_queries_per_task": 100,
            "new_temperatures": 2,
            "schedules_per_temperature": 3,
            "rollouts_per_query_cell": FORMAL_ROLLOUTS,
            "new_generations": EXPECTED_FORMAL_GENERATIONS,
            "num_shards": DEFAULT_NUM_SHARDS,
            "generations_per_shard": EXPECTED_FORMAL_GENERATIONS // DEFAULT_NUM_SHARDS,
        },
        "analysis_rules": {
            "parser_failures_and_empty_responses": "retain as incorrect",
            "missing_or_invalid_artifact": "block analysis",
            "query_exclusions_after_outcome_read": False,
            "first_correctness_read_must_be_logged": True,
            "secondary_endpoints_change_primary_verdict": False,
        },
        "not_permitted": [
            "calling this follow-up part of the original Gate-1 chain",
            "claiming temperature and schedule are universally independent",
            "claiming equivalence from an interval that includes zero",
            "collapsing crossing endpoints into a single coverage AUC verdict",
            "adding a temperature, endpoint, policy, task, or exclusion after outcome read",
        ],
    }


def emit_freeze(*, output: Path, repo_root: Path, control_inventory: Path) -> str:
    if output.exists():
        raise TemperatureExtensionError(f"refusing to overwrite freeze: {output}")
    _require_clean_repo(repo_root)
    inventory = json.loads(control_inventory.read_text(encoding="utf-8"))
    if (
        inventory.get("inventory_version") != CONTROL_INVENTORY_VERSION
        or inventory.get("split_manifest_sha256") != SPLIT_MANIFEST_SHA256
        or inventory.get("artifact_count") != EXPECTED_CONTROL_ARTIFACTS
        or inventory.get("outcome_fields_read") is not False
    ):
        raise TemperatureExtensionError("control inventory is not freeze-eligible")
    payload = build_freeze_payload(
        runner_commit=_repo_commit(repo_root),
        control_inventory_sha256=sha256_file(control_inventory),
    )
    write_json_atomic(output, payload)
    validate_freeze(output, runner_commit=payload["runner_commit"])
    return sha256_file(output)


def _schema(path: Path) -> Draft202012Validator:
    if sha256_file(path) != ARTIFACT_SCHEMA_SHA256:
        raise TemperatureExtensionError("temperature artifact-schema hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(payload)
    return Draft202012Validator(payload)


def _policy_payload(policy_name: str) -> dict[str, Any]:
    policy = POLICY_BY_NAME[policy_name]
    config = policy.decode_config
    return {
        "name": policy.name,
        "schedule": policy.schedule,
        "steps": config.steps,
        "generation_length": config.gen_length,
        "block_length": config.block_length,
        "temperature": config.temperature,
        "cfg_scale": config.cfg_scale,
        "remasking": config.remasking,
    }


def _hash_keys(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def validate_new_artifacts(
    *,
    temperature_root: Path,
    manifest: dict[str, Any],
    freeze_path: Path,
    schema_path: Path,
    run_kind: str,
) -> dict[str, Any]:
    freeze_sha = validate_freeze(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validator = _schema(schema_path)
    items = build_work_items(manifest, run_kind=run_kind)
    missing: list[str] = []
    invalid: list[str] = []
    for item in items:
        path = _new_result_path(
            temperature_root,
            run_kind,
            item.task,
            item.policy_name,
            item.row_index,
            item.rollout_index,
        )
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            validator.validate(record)
            expected = {
                "protocol_version": PROTOCOL_VERSION,
                "run_kind": run_kind,
                "temperature_extension_freeze_sha256": freeze_sha,
                "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "external_repository_commit": JUSTGRPO_COMMIT,
                "rollout_index": item.rollout_index,
                "seed_key": _seed_key(item.task, item.row_index, item.rollout_index),
                "policy": _policy_payload(item.policy_name),
                "completion_status": "complete",
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise TemperatureExtensionError("record provenance mismatch")
            if record.get("runner_repository") != {
                "commit": freeze["runner_commit"],
                "dirty": False,
            }:
                raise TemperatureExtensionError("runner commit mismatch")
            source = record["source"]
            expected_source = manifest["sources"][item.task]
            if (
                source["row_index"] != item.row_index
                or source["split_role"] != item.split_role
                or source["prompt_sha256"] != item.prompt_sha256
                or source["dataset_id"] != expected_source["dataset_id"]
                or source["dataset_revision"] != expected_source["revision"]
                or source["dataset_split"] != expected_source["split"]
            ):
                raise TemperatureExtensionError("source mismatch")
            trace = _trace_path(path)
            if not trace.is_file() or sha256_file(trace) != record["trace_sha256"]:
                raise TemperatureExtensionError("trace mismatch")
        except Exception as error:
            invalid.append(f"{path}: {error}")

    num_shards = 1 if run_kind == "temperature-smoke" else DEFAULT_NUM_SHARDS
    marker_errors: list[str] = []
    marker_results = 0
    for shard in range(num_shards):
        path = (
            temperature_root
            / PROTOCOL_VERSION
            / "manifests"
            / run_kind
            / "all"
            / f"shard-{shard:03d}-of-{num_shards:03d}.json"
        )
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            validator.validate(marker)
            expected_items = build_work_items(
                manifest,
                run_kind=run_kind,
                shard_index=shard,
                num_shards=num_shards,
            )
            expected = {
                "protocol_version": PROTOCOL_VERSION,
                "run_kind": run_kind,
                "task_selection": "all",
                "shard_index": shard,
                "num_shards": num_shards,
                "expected_keys_sha256": _hash_keys(
                    [item.key for item in expected_items]
                ),
                "observed_keys_sha256": _hash_keys(
                    [item.key for item in expected_items]
                ),
                "result_count": len(expected_items),
            }
            if any(marker.get(key) != value for key, value in expected.items()):
                raise TemperatureExtensionError("marker mismatch")
            marker_results += marker["result_count"]
        except Exception as error:
            marker_errors.append(f"{path}: {error}")
    return {
        "expected": len(items),
        "present_and_valid": len(items) - len(missing) - len(invalid),
        "missing": len(missing),
        "invalid": len(invalid),
        "missing_examples": missing[:10],
        "invalid_examples": invalid[:10],
        "markers": num_shards - len(marker_errors),
        "marker_results": marker_results,
        "marker_errors": marker_errors[:10],
        "outcome_fields_read": False,
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _pass(flags: list[bool], k: int) -> float:
    return estimate_pass_at_k(len(flags), sum(flags), k)


def _high(flags: list[bool]) -> float:
    return statistics.mean(_pass(flags, k) for k in HIGH_BUDGET_GRID)


def _answer_entropy(answers: list[Any]) -> float:
    counts = Counter(answer for answer in answers if answer is not None)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _fractional_majority_correct(records: list[dict[str, Any]], k: int) -> float:
    selected = records[:k]
    counts = Counter(
        record["parsed_answer"]
        for record in selected
        if record["parsed_answer"] is not None
    )
    if not counts:
        return 0.0
    maximum = max(counts.values())
    modes = [answer for answer, count in counts.items() if count == maximum]
    answer_correctness = {
        answer: any(
            bool(record["correct"]) and record["parsed_answer"] == answer
            for record in selected
        )
        for answer in modes
    }
    return statistics.mean(float(answer_correctness[answer]) for answer in modes)


def _simultaneous_primary(values: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    endpoint_names = [
        f"t{suffix}_{level}" for suffix in ("09", "12") for level in ("low", "high")
    ]
    points = {
        name: statistics.mean(statistics.mean(values[name][task]) for task in TASKS)
        for name in endpoint_names
    }
    rng = random.Random(BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {name: [] for name in endpoint_names}
    max_deviations: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = {}
        task_indices = {
            task: [
                rng.randrange(len(values[endpoint_names[0]][task]))
                for _ in values[endpoint_names[0]][task]
            ]
            for task in TASKS
        }
        for name in endpoint_names:
            task_means = [
                statistics.mean(
                    values[name][task][index] for index in task_indices[task]
                )
                for task in TASKS
            ]
            draw[name] = statistics.mean(task_means)
            draws[name].append(draw[name])
        max_deviations.append(
            max(abs(draw[name] - points[name]) for name in endpoint_names)
        )
    critical = _percentile(max_deviations, 0.95)
    endpoints = {}
    for name in endpoint_names:
        endpoints[name] = {
            "point": points[name],
            "raw_95_ci": [
                _percentile(draws[name], 0.025),
                _percentile(draws[name], 0.975),
            ],
            "simultaneous_95_ci": [points[name] - critical, points[name] + critical],
            "per_task_points": {
                task: statistics.mean(values[name][task]) for task in TASKS
            },
        }
    verdicts = {}
    for suffix in ("09", "12"):
        low = endpoints[f"t{suffix}_low"]["simultaneous_95_ci"]
        high = endpoints[f"t{suffix}_high"]["simultaneous_95_ci"]
        verdicts[f"t{suffix}"] = (
            "resolved_reversal"
            if low[1] < 0 and high[0] > 0
            else "not_resolved_as_reversal"
        )
    return {
        "family_size": 4,
        "simultaneous_method": "nonstudentized maximum absolute deviation",
        "simultaneous_critical_value": critical,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "endpoints": endpoints,
        "verdicts": verdicts,
    }


def analyze(
    *,
    phase1_root: Path,
    temperature_root: Path,
    manifest: dict[str, Any],
    freeze_path: Path,
    schema_path: Path,
    control_inventory: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    control_validation = validate_control_inventory(
        phase1_root=phase1_root,
        inventory_path=control_inventory,
        expected_sha256=freeze["reused_control_inventory_sha256"],
    )
    new_validation = validate_new_artifacts(
        temperature_root=temperature_root,
        manifest=manifest,
        freeze_path=freeze_path,
        schema_path=schema_path,
        run_kind="temperature-formal",
    )
    if (
        new_validation["present_and_valid"] != EXPECTED_FORMAL_GENERATIONS
        or new_validation["markers"] != DEFAULT_NUM_SHARDS
    ):
        raise TemperatureExtensionError("formal artifact closure failed")

    flags: dict[tuple[str, int, str, float], list[bool]] = {}
    records_by_cell: dict[tuple[str, int, str, float], list[dict[str, Any]]] = {}
    for task, queries in _queries(manifest, "heldout").items():
        for query in queries:
            row = int(query["row_index"])
            for schedule in CONTROL_POLICY_BY_SCHEDULE:
                records = [
                    json.loads(
                        _control_result_path(
                            phase1_root, task, row, schedule, rollout
                        ).read_text(encoding="utf-8")
                    )
                    for rollout in CONTROL_ROLLOUTS
                ]
                flags[(task, row, schedule, 0.6)] = [
                    bool(record["correct"]) for record in records
                ]
                records_by_cell[(task, row, schedule, 0.6)] = records
            for policy in TEMPERATURE_POLICIES:
                records = [
                    json.loads(
                        _new_result_path(
                            temperature_root,
                            "temperature-formal",
                            task,
                            policy.name,
                            row,
                            rollout,
                        ).read_text(encoding="utf-8")
                    )
                    for rollout in range(FORMAL_ROLLOUTS)
                ]
                flags[
                    (task, row, policy.schedule, policy.decode_config.temperature)
                ] = [bool(record["correct"]) for record in records]
                records_by_cell[
                    (task, row, policy.schedule, policy.decode_config.temperature)
                ] = records

    primary_values = {
        name: {task: [] for task in TASKS}
        for name in ("t09_low", "t09_high", "t12_low", "t12_high")
    }
    for temperature, suffix in ((0.9, "09"), (1.2, "12")):
        for task, queries in _queries(manifest, "heldout").items():
            for query in queries:
                row = int(query["row_index"])
                rnd = flags[(task, row, "random", temperature)]
                cf = flags[(task, row, "confidence_first", temperature)]
                primary_values[f"t{suffix}_low"][task].append(
                    _pass(rnd, 1) - _pass(cf, 1)
                )
                primary_values[f"t{suffix}_high"][task].append(_high(rnd) - _high(cf))

    curves: dict[str, Any] = {}
    for temperature in (0.6, *TEMPERATURES):
        temperature_key = f"{temperature:.1f}"
        curves[temperature_key] = {}
        for schedule in CONTROL_POLICY_BY_SCHEDULE:
            curves[temperature_key][schedule] = {}
            for task, queries in _queries(manifest, "heldout").items():
                task_curves = {
                    str(k): statistics.mean(
                        _pass(
                            flags[
                                (task, int(query["row_index"]), schedule, temperature)
                            ],
                            k,
                        )
                        for query in queries
                    )
                    for k in K_GRID
                }
                curves[temperature_key][schedule][task] = task_curves
            curves[temperature_key][schedule]["equal_task_pooled"] = {
                str(k): statistics.mean(
                    curves[temperature_key][schedule][task][str(k)] for task in TASKS
                )
                for k in K_GRID
            }

    within_temperature_contrasts: dict[str, Any] = {}
    for temperature in (0.6, *TEMPERATURES):
        temperature_key = f"{temperature:.1f}"
        within_temperature_contrasts[temperature_key] = {}
        for left, right in (
            ("random", "confidence_first"),
            ("confidence_first", "sequential"),
            ("random", "sequential"),
        ):
            name = f"{left}_minus_{right}"
            within_temperature_contrasts[temperature_key][name] = {
                str(k): (
                    curves[temperature_key][left]["equal_task_pooled"][str(k)]
                    - curves[temperature_key][right]["equal_task_pooled"][str(k)]
                )
                for k in K_GRID
            }

    cross_temperature_frontier: dict[str, Any] = {}
    for schedule in CONTROL_POLICY_BY_SCHEDULE:
        for temperature in TEMPERATURES:
            new_key = f"{temperature:.1f}"
            name = f"{schedule}_t{new_key}_minus_t0.6"
            cross_temperature_frontier[name] = {
                "pass_at_1": (
                    curves[new_key][schedule]["equal_task_pooled"]["1"]
                    - curves["0.6"][schedule]["equal_task_pooled"]["1"]
                ),
                "high_budget": statistics.mean(
                    curves[new_key][schedule]["equal_task_pooled"][str(k)]
                    - curves["0.6"][schedule]["equal_task_pooled"][str(k)]
                    for k in HIGH_BUDGET_GRID
                ),
            }
    for temperature in TEMPERATURES:
        new_key = f"{temperature:.1f}"
        cross_temperature_frontier[f"cf_t{new_key}_minus_random_t0.6"] = {
            "pass_at_1": (
                curves[new_key]["confidence_first"]["equal_task_pooled"]["1"]
                - curves["0.6"]["random"]["equal_task_pooled"]["1"]
            ),
            "high_budget": statistics.mean(
                curves[new_key]["confidence_first"]["equal_task_pooled"][str(k)]
                - curves["0.6"]["random"]["equal_task_pooled"][str(k)]
                for k in HIGH_BUDGET_GRID
            ),
        }

    diagnostics: dict[str, Any] = {}
    for temperature in (0.6, *TEMPERATURES):
        temperature_key = f"{temperature:.1f}"
        diagnostics[temperature_key] = {}
        for schedule in CONTROL_POLICY_BY_SCHEDULE:
            cell_records = [
                records_by_cell[(task, int(query["row_index"]), schedule, temperature)]
                for task, queries in _queries(manifest, "heldout").items()
                for query in queries
            ]
            flattened = [record for records in cell_records for record in records]
            diagnostics[temperature_key][schedule] = {
                "quality": {
                    "record_count": len(flattened),
                    "empty_response_rate": statistics.mean(
                        not str(record["response"]).strip() for record in flattened
                    ),
                    "parser_error_rate": statistics.mean(
                        record["parser_status"] != "ok" for record in flattened
                    ),
                    "mean_whitespace_tokens": statistics.mean(
                        len(str(record["response"]).split()) for record in flattened
                    ),
                    "mean_wall_time_seconds": statistics.mean(
                        float(record["wall_time_seconds"]) for record in flattened
                    ),
                },
                "by_k": {},
            }
            for k in (1, 2, 4, 8, 16, 32):
                diagnostics[temperature_key][schedule]["by_k"][str(k)] = {
                    "majority_vote_accuracy_ties_fractional": statistics.mean(
                        _fractional_majority_correct(records, k)
                        for records in cell_records
                    ),
                    "mean_unique_parsed_answers": statistics.mean(
                        len(
                            {
                                record["parsed_answer"]
                                for record in records[:k]
                                if record["parsed_answer"] is not None
                            }
                        )
                        for records in cell_records
                    ),
                    "mean_parsed_answer_entropy": statistics.mean(
                        _answer_entropy(
                            [record["parsed_answer"] for record in records[:k]]
                        )
                        for records in cell_records
                    ),
                }

    return {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha256_file(freeze_path),
        "evidence_status": "post-review prospective follow-up",
        "validation": {
            "controls": control_validation,
            "new_artifacts": new_validation,
        },
        "primary": _simultaneous_primary(primary_values),
        "secondary_curves": curves,
        "secondary_within_temperature_contrasts": within_temperature_contrasts,
        "secondary_cross_temperature_frontier": cross_temperature_frontier,
        "secondary_diagnostics": diagnostics,
        "interpretation_boundary": (
            "Primary verdicts concern only whether the RND-CF low/high sign reversal "
            "replicates at T=0.9 or T=1.2 for this model and these two tasks."
        ),
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Temperature-by-Schedule Extension Results",
        "",
        "**Evidence status:** post-review prospective follow-up; not part of the original Gate-1 chain.",
        "",
        "## Artifact closure",
        "",
        f"- Reused T=0.6 controls: {report['validation']['controls']['present_and_hash_matched']}/19200.",
        f"- New records: {report['validation']['new_artifacts']['present_and_valid']}/38400.",
        f"- Formal shard markers: {report['validation']['new_artifacts']['markers']}/6.",
        "",
        "## Frozen primary family",
        "",
        "| Temperature | Endpoint | RND - CF | Simultaneous 95% CI |",
        "|---:|---|---:|---:|",
    ]
    endpoints = report["primary"]["endpoints"]
    for suffix, temperature in (("09", "0.9"), ("12", "1.2")):
        for level in ("low", "high"):
            result = endpoints[f"t{suffix}_{level}"]
            interval = result["simultaneous_95_ci"]
            lines.append(
                f"| {temperature} | {level} | {result['point']:+.5f} | "
                f"[{interval[0]:+.5f}, {interval[1]:+.5f}] |"
            )
        lines.append(
            f"| {temperature} | frozen verdict | "
            f"`{report['primary']['verdicts'][f't{suffix}']}` |  |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            report["interpretation_boundary"],
            "Secondary curves cannot modify the frozen primary verdicts.",
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
        choices=[
            "build-control-inventory",
            "emit-freeze",
            "validate-smoke",
            "validate-only",
            "analyze",
        ],
        required=True,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--control-inventory", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--temperature-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--invocation-log", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_split_manifest(args.split_manifest)
    if args.mode == "build-control-inventory":
        if args.phase1_root is None:
            raise TemperatureExtensionError("inventory requires --phase1-root")
        if args.invocation_log is not None:
            _append_invocation(
                args.invocation_log,
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": args.mode,
                    "reads_temperature_correctness": False,
                    "reads_reused_control_correctness": False,
                    "analyzer_commit": _repo_commit(args.repo_root),
                },
            )
        print(
            json.dumps(
                build_control_inventory(
                    phase1_root=args.phase1_root,
                    manifest=manifest,
                    output=args.control_inventory,
                ),
                indent=2,
            )
        )
        return 0
    if args.mode == "emit-freeze":
        if args.freeze is None:
            raise TemperatureExtensionError("emit-freeze requires --freeze")
        freeze_sha = emit_freeze(
            output=args.freeze,
            repo_root=args.repo_root,
            control_inventory=args.control_inventory,
        )
        print(json.dumps({"freeze": str(args.freeze), "sha256": freeze_sha}, indent=2))
        return 0

    if args.freeze is None or args.temperature_root is None:
        raise TemperatureExtensionError(
            "validation/analysis require freeze and temperature root"
        )
    event = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "reads_temperature_correctness": args.mode == "analyze",
        "freeze_sha256": sha256_file(args.freeze),
        "analyzer_commit": _repo_commit(args.repo_root),
    }
    if args.invocation_log is not None:
        _append_invocation(args.invocation_log, event)

    run_kind = (
        "temperature-smoke" if args.mode == "validate-smoke" else "temperature-formal"
    )
    if args.mode in {"validate-smoke", "validate-only"}:
        validation = validate_new_artifacts(
            temperature_root=args.temperature_root,
            manifest=manifest,
            freeze_path=args.freeze,
            schema_path=args.artifact_schema,
            run_kind=run_kind,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    if args.phase1_root is None or args.output_json is None or args.output_md is None:
        raise TemperatureExtensionError(
            "analysis requires --phase1-root, --output-json, and --output-md"
        )
    report = analyze(
        phase1_root=args.phase1_root,
        temperature_root=args.temperature_root,
        manifest=manifest,
        freeze_path=args.freeze,
        schema_path=args.artifact_schema,
        control_inventory=args.control_inventory,
    )
    write_json_atomic(args.output_json, report)
    write_text_atomic(args.output_md, render_report(report))
    print(json.dumps(report["primary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
