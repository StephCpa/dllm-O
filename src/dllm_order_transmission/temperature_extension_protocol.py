"""Immutable planning and provenance checks for the temperature extension."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .decoding import DecodeConfig
from .phase1_protocol import (
    JUSTGRPO_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_SEED,
    SPLIT_MANIFEST_SHA256,
)


PROTOCOL_VERSION = "temperature-extension-v1"
FREEZE_VERSION = "temperature-extension-freeze-v1"
ARTIFACT_SCHEMA_SHA256 = (
    "aff8a7d8417171cd80484dd6179981bd4e51a9816edfb48848a8a7eb5c50dc28"
)
TEMPERATURES = (0.9, 1.2)
FORMAL_ROLLOUTS = 32
DEFAULT_NUM_SHARDS = 6
EXPECTED_FORMAL_GENERATIONS = 38_400
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPLICATES = 10_000
K_GRID = (1, 2, 4, 8, 16, 32)
HIGH_BUDGET_GRID = (16, 32)


@dataclass(frozen=True)
class TemperaturePolicy:
    name: str
    schedule: str
    decode_config: DecodeConfig


@dataclass(frozen=True)
class TemperatureWorkItem:
    task: str
    split_role: str
    run_kind: str
    row_index: int
    prompt_sha256: str
    policy_name: str
    rollout_index: int

    @property
    def key(self) -> str:
        return (
            f"{PROTOCOL_VERSION}|{self.task}|{self.split_role}|{self.run_kind}|"
            f"q{self.row_index:04d}|{self.policy_name}|r{self.rollout_index:02d}"
        )


def _policy(
    name: str,
    schedule: str,
    *,
    block_length: int,
    temperature: float,
    remasking: str,
) -> TemperaturePolicy:
    return TemperaturePolicy(
        name=name,
        schedule=schedule,
        decode_config=DecodeConfig(
            steps=256,
            gen_length=256,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=0.0,
            remasking=remasking,
            mask_id=126336,
        ),
    )


TEMPERATURE_POLICIES = tuple(
    policy
    for temperature, suffix in ((0.9, "09"), (1.2, "12"))
    for policy in (
        _policy(
            f"sequential_b1_t{suffix}",
            "sequential",
            block_length=1,
            temperature=temperature,
            remasking="low_confidence",
        ),
        _policy(
            f"cf_b32_t{suffix}",
            "confidence_first",
            block_length=32,
            temperature=temperature,
            remasking="low_confidence",
        ),
        _policy(
            f"rnd_b32_t{suffix}",
            "random",
            block_length=32,
            temperature=temperature,
            remasking="random",
        ),
    )
)
POLICY_BY_NAME = {policy.name: policy for policy in TEMPERATURE_POLICIES}


def build_work_items(
    manifest: dict[str, Any],
    *,
    run_kind: str,
    task_selection: str = "all",
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[TemperatureWorkItem]:
    if run_kind not in {"temperature-smoke", "temperature-formal"}:
        raise ValueError("unknown temperature-extension run kind")
    if task_selection not in {"all", "gsm8k", "math500"}:
        raise ValueError("unknown task selection")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num_shards")

    tasks = ("gsm8k", "math500") if task_selection == "all" else (task_selection,)
    split_role = "calibration" if run_kind == "temperature-smoke" else "heldout"
    rollout_indices = range(1) if run_kind == "temperature-smoke" else range(32)
    all_items: list[TemperatureWorkItem] = []
    for task in tasks:
        entries = manifest["splits"][task][split_role]
        if run_kind == "temperature-smoke":
            entries = entries[:1]
        for entry in entries:
            for policy in TEMPERATURE_POLICIES:
                for rollout_index in rollout_indices:
                    all_items.append(
                        TemperatureWorkItem(
                            task=task,
                            split_role=split_role,
                            run_kind=run_kind,
                            row_index=int(entry["row_index"]),
                            prompt_sha256=str(entry["prompt_sha256"]),
                            policy_name=policy.name,
                            rollout_index=rollout_index,
                        )
                    )
    return [
        item
        for index, item in enumerate(all_items)
        if index % num_shards == shard_index
    ]


def validate_freeze(path: Path, *, runner_commit: str | None = None) -> str:
    if not path.is_file():
        raise RuntimeError("missing temperature-extension freeze")
    freeze_sha = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version": FREEZE_VERSION,
        "selection_status": "post-review prospective follow-up",
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "artifact_schema_sha256": ARTIFACT_SCHEMA_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "external_repository_commit": JUSTGRPO_COMMIT,
        "protocol_seed": PROTOCOL_SEED,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "temperature-extension freeze mismatch: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    inventory_sha = payload.get("reused_control_inventory_sha256", "")
    if len(inventory_sha) != 64 or any(
        character not in "0123456789abcdef" for character in inventory_sha
    ):
        raise RuntimeError("freeze does not bind the reused T=0.6 control inventory")
    if payload.get("new_temperatures") != list(TEMPERATURES):
        raise RuntimeError("freeze uses another temperature grid")
    if payload.get("new_policy_names") != [
        policy.name for policy in TEMPERATURE_POLICIES
    ]:
        raise RuntimeError("freeze uses another policy grid")
    cardinality = payload.get("cardinality", {})
    expected_cardinality = {
        "tasks": ["gsm8k", "math500"],
        "heldout_queries_per_task": 100,
        "new_temperatures": 2,
        "schedules_per_temperature": 3,
        "rollouts_per_query_cell": FORMAL_ROLLOUTS,
        "new_generations": EXPECTED_FORMAL_GENERATIONS,
        "num_shards": DEFAULT_NUM_SHARDS,
        "generations_per_shard": EXPECTED_FORMAL_GENERATIONS // DEFAULT_NUM_SHARDS,
    }
    if cardinality != expected_cardinality:
        raise RuntimeError("freeze uses another experiment cardinality")
    if runner_commit is not None and payload.get("runner_commit") != runner_commit:
        raise RuntimeError("temperature-extension freeze is bound to another commit")
    if payload.get("analyzer_commit") != payload.get("runner_commit"):
        raise RuntimeError("temperature analyzer and runner must share a frozen commit")
    return freeze_sha
