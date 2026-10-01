"""Immutable work planning and provenance checks for Condition A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .decoding import DecodeConfig
from .phase1_protocol import (
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_SEED,
    SPLIT_MANIFEST_SHA256,
)


PROTOCOL_VERSION = "condition-a-v1"
FREEZE_VERSION = "condition-a-freeze-v1"
ARTIFACT_SCHEMA_SHA256 = (
    "4a8add307ec2d252eb814e7d332eb5cb48ac3a2e858cef4bc0bdd1e1bb90e474"
)
OPTION_B_FREEZE_SHA256 = (
    "b83edeff717e8bfbf59feea5f38dc6e266d50fbcd42a1aa440fe510daf8f5126"
)
OPTION_B_REPORT_SHA256 = (
    "dbcf3ae1a12fc8f47d575922fd5e1b24eb2b004433b96f832da5e8e0b4e08c3f"
)
OPTION_B_REPAIR_AMENDMENT_SHA256 = (
    "5f1f32523aac816e0f21bad690a7561a8f8df01698efa4672858042d42841673"
)
POLICY_NAME = "ef_b32"
FORMAL_ROLLOUTS = 64
EXPECTED_FORMAL_GENERATIONS = 12_800
DEFAULT_NUM_SHARDS = 8


@dataclass(frozen=True)
class ConditionAWorkItem:
    task: str
    split_role: str
    run_kind: str
    row_index: int
    prompt_sha256: str
    rollout_index: int

    @property
    def key(self) -> str:
        return (
            f"{PROTOCOL_VERSION}|{self.task}|{self.split_role}|{self.run_kind}|"
            f"q{self.row_index:04d}|{POLICY_NAME}|r{self.rollout_index:02d}"
        )


CONDITION_A_POLICY = DecodeConfig(
    steps=256,
    gen_length=256,
    block_length=32,
    temperature=0.6,
    cfg_scale=0.0,
    remasking="high_entropy",
    mask_id=126336,
)


def build_work_items(
    manifest: dict[str, Any],
    *,
    run_kind: str,
    task_selection: str = "all",
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[ConditionAWorkItem]:
    if run_kind not in {"condition-a-smoke", "condition-a-formal"}:
        raise ValueError("unknown Condition A run kind")
    if task_selection not in {"all", "gsm8k", "math500"}:
        raise ValueError("unknown task selection")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num_shards")

    tasks = ("gsm8k", "math500") if task_selection == "all" else (task_selection,)
    split_role = "calibration" if run_kind == "condition-a-smoke" else "heldout"
    rollout_indices = range(1) if run_kind == "condition-a-smoke" else range(64)
    all_items: list[ConditionAWorkItem] = []
    for task in tasks:
        entries = manifest["splits"][task][split_role]
        if run_kind == "condition-a-smoke":
            entries = entries[:1]
        for entry in entries:
            for rollout_index in rollout_indices:
                all_items.append(
                    ConditionAWorkItem(
                        task=task,
                        split_role=split_role,
                        run_kind=run_kind,
                        row_index=int(entry["row_index"]),
                        prompt_sha256=str(entry["prompt_sha256"]),
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
        raise RuntimeError("missing Condition A freeze")
    freeze_sha = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version": FREEZE_VERSION,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "artifact_schema_sha256": ARTIFACT_SCHEMA_SHA256,
        "option_b_freeze_sha256": OPTION_B_FREEZE_SHA256,
        "option_b_report_sha256": OPTION_B_REPORT_SHA256,
        "option_b_repair_amendment_sha256": OPTION_B_REPAIR_AMENDMENT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Condition A freeze mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )
    if payload.get("trigger", {}).get("run_condition_a") is not True:
        raise RuntimeError("Condition A freeze does not record a positive gate")
    if runner_commit is not None and payload.get("runner_commit") != runner_commit:
        raise RuntimeError("Condition A freeze is bound to another runner commit")
    return freeze_sha
