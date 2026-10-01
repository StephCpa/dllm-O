"""Immutable Phase 1 experiment configuration and work-key planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .decoding import DecodeConfig


PROTOCOL_VERSION = "phase1-v1"
PROTOCOL_SEED = 20260823
SPLIT_MANIFEST_SHA256 = (
    "68e4343846e2a8c7c3f5fba12a2cca9cbc2866f316e4fdb49adaf7cd40243b9b"
)
ARTIFACT_SCHEMA_SHA256 = (
    "8f81eb5736c3a9702cb9c155ebac40516fd08a3304ae72823633821710d53bb3"
)
PHASE2_ARTIFACT_SCHEMA_SHA256 = (
    "f3ea4bb30c4ff22fd573cf2c02bdbd2e81d44424cd748780bdd878915134fad7"
)
RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256 = (
    "1f9e06530993ea89b972720671e96bdca8bbc348237a0989b44bc2127b8362bf"
)
BCI_FREEZE_SHA256 = (
    "8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17"
)
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
JUSTGRPO_COMMIT = "1a2fddb5c6655597e63081c0af5ebb718a849f39"


@dataclass(frozen=True)
class Phase1Policy:
    name: str
    decode_config: DecodeConfig


@dataclass(frozen=True)
class Phase1WorkItem:
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


def _policy(name: str, block_length: int, remasking: str) -> Phase1Policy:
    return Phase1Policy(
        name=name,
        decode_config=DecodeConfig(
            steps=256,
            gen_length=256,
            block_length=block_length,
            temperature=0.6,
            cfg_scale=0.0,
            remasking=remasking,
            mask_id=126336,
        ),
    )


PHASE1_POLICIES = (
    _policy("ar_b1", 1, "low_confidence"),
    _policy("ao_b8", 8, "low_confidence"),
    _policy("ao_b16", 16, "low_confidence"),
    _policy("ao_b32", 32, "low_confidence"),
    _policy("random_b32", 32, "random"),
)
POLICY_BY_NAME = {policy.name: policy for policy in PHASE1_POLICIES}
PRIMARY_POLICY_NAMES = ("ar_b1", "ao_b32")
PHASE2_POLICY_NAMES = ("ao_b8", "ao_b16")


def load_split_manifest(path: Path) -> dict[str, Any]:
    observed_hash = sha256_file(path)
    if observed_hash != SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            "Phase 1 split-manifest hash mismatch: "
            f"expected {SPLIT_MANIFEST_SHA256}, found {observed_hash}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split_version") != "phase1-splits-v1":
        raise RuntimeError("unexpected Phase 1 split-manifest version")
    for task in ("gsm8k", "math500"):
        split = manifest.get("splits", {}).get(task, {})
        if len(split.get("calibration", [])) != 50:
            raise RuntimeError(f"{task}: expected 50 calibration rows")
        if len(split.get("heldout", [])) != 100:
            raise RuntimeError(f"{task}: expected 100 held-out rows")
        calibration = {row["row_index"] for row in split["calibration"]}
        heldout = {row["row_index"] for row in split["heldout"]}
        if calibration & heldout:
            raise RuntimeError(f"{task}: calibration and held-out rows overlap")
    return manifest


def validate_bci_freeze(path: Path, split_manifest_sha256: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "created_at_utc",
        "split_manifest_sha256",
        "calibration_artifact_hashes",
        "candidate_boundary_rule",
        "matching_rule",
        "bci_formula",
        "baseline_predictor",
        "augmented_predictor",
        "normalization",
        "parser_policy",
        "exclusion_policy",
        "target_policy",
        "failure_thresholds",
        "bootstrap_seed",
        "runner_commit",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"BCI freeze is missing required fields: {missing}")
    if payload["protocol_version"] != "phase1-bci-freeze-v1":
        raise RuntimeError("unexpected BCI-freeze protocol version")
    if payload["split_manifest_sha256"] != split_manifest_sha256:
        raise RuntimeError("BCI freeze is bound to a different split manifest")
    if not payload["calibration_artifact_hashes"]:
        raise RuntimeError("BCI freeze must bind at least one calibration artifact")
    return sha256_file(path)


def validate_phase2_freeze(path: Path, split_manifest_sha256: str) -> str:
    """Validate the Phase 2 Option B freeze structure and return its SHA-256."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "created_at_utc",
        "split_manifest_sha256",
        "bci_freeze_sha256",
        "runner_commit",
        "phase2_policies",
        "rollout_budget",
        "primary_endpoints",
        "verdict_taxonomy",
        "condition_a_gate",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"Phase 2 freeze is missing required fields: {missing}")
    if payload["protocol_version"] != "phase2-optionb-freeze-v1":
        raise RuntimeError("unexpected Phase 2 freeze protocol version")
    if payload["split_manifest_sha256"] != split_manifest_sha256:
        raise RuntimeError("Phase 2 freeze is bound to a different split manifest")
    if payload["bci_freeze_sha256"] != BCI_FREEZE_SHA256:
        raise RuntimeError("Phase 2 freeze must reference the final BCI freeze")
    return sha256_file(path)


def validate_random_extension_freeze(
    path: Path, split_manifest_sha256: str
) -> str:
    """Validate the post-Gate random-policy extension freeze."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "created_at_utc",
        "selection_status",
        "split_manifest_sha256",
        "bci_freeze_sha256",
        "artifact_schema_sha256",
        "runner_commit",
        "policy",
        "rollout_budget",
        "primary_endpoint",
        "secondary_endpoints",
        "verdict_taxonomy",
        "cardinality",
        "analysis_rules",
        "not_permitted",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(
            f"random-extension freeze is missing required fields: {missing}"
        )
    if payload["protocol_version"] != "random-extension-freeze-v1":
        raise RuntimeError("unexpected random-extension freeze protocol version")
    expected_status = "post-unblinding targeted prospective follow-up"
    if payload["selection_status"] != expected_status:
        raise RuntimeError("random extension must disclose post-unblinding selection")
    if payload["split_manifest_sha256"] != split_manifest_sha256:
        raise RuntimeError("random-extension freeze uses another split manifest")
    if payload["bci_freeze_sha256"] != BCI_FREEZE_SHA256:
        raise RuntimeError("random-extension freeze must bind the final BCI freeze")
    if (
        payload["artifact_schema_sha256"]
        != RANDOM_EXTENSION_ARTIFACT_SCHEMA_SHA256
    ):
        raise RuntimeError("random-extension freeze uses another artifact schema")
    if payload["policy"] != "random_b32":
        raise RuntimeError("random-extension freeze must use random_b32")
    rollout_budget = payload["rollout_budget"]
    if rollout_budget != {
        "existing_indices": [0, 15],
        "new_indices": [16, 63],
        "analysis_rollouts_per_query": 64,
    }:
        raise RuntimeError("random-extension freeze uses another rollout budget")
    primary = payload["primary_endpoint"]
    if (
        primary.get("name") != "HighBudgetCoverage64"
        or primary.get("definition")
        != "mean unbiased Pass@k over k in {16,32,64}"
        or primary.get("contrast") != "random_b32 - ao_b32"
        or primary.get("unit") != "paired held-out query"
    ):
        raise RuntimeError("random-extension freeze uses another primary endpoint")
    cardinality = payload["cardinality"]
    if (
        cardinality.get("tasks") != ["gsm8k", "math500"]
        or cardinality.get("heldout_queries_per_task") != 100
        or cardinality.get("new_rollouts_per_query") != 48
        or cardinality.get("new_generations") != 9_600
        or cardinality.get("num_shards") != 3
        or cardinality.get("generations_per_shard") != 3_200
    ):
        raise RuntimeError("random-extension freeze uses another cardinality")
    return sha256_file(path)


def _policy_names(task: str, split_role: str, run_kind: str) -> tuple[str, ...]:
    if run_kind == "smoke":
        if split_role != "calibration":
            raise ValueError("smoke must use calibration rows")
        return tuple(POLICY_BY_NAME)
    if run_kind == "screening":
        if split_role == "calibration" and task == "gsm8k":
            return ("ao_b8", "ao_b16", "random_b32")
        return tuple(POLICY_BY_NAME)
    if run_kind == "primary-extension":
        if split_role != "heldout":
            raise ValueError("primary extension is held-out only")
        return PRIMARY_POLICY_NAMES
    if run_kind == "phase2-extension":
        if split_role != "heldout":
            raise ValueError("phase2 extension is held-out only")
        return PHASE2_POLICY_NAMES
    if run_kind == "random-extension":
        if split_role != "heldout":
            raise ValueError("random extension is held-out only")
        return ("random_b32",)
    raise ValueError(f"unknown run kind: {run_kind}")


def _rollout_indices(run_kind: str) -> range:
    if run_kind == "smoke":
        return range(1)
    if run_kind == "screening":
        return range(16)
    if run_kind in ("primary-extension", "phase2-extension", "random-extension"):
        return range(16, 64)
    raise ValueError(f"unknown run kind: {run_kind}")


def build_phase1_work_items(
    manifest: dict[str, Any],
    *,
    split_role: str,
    run_kind: str,
    task_selection: str = "all",
    shard_index: int = 0,
    num_shards: int = 1,
    policies: tuple[str, ...] | None = None,
) -> list[Phase1WorkItem]:
    if split_role not in {"calibration", "heldout"}:
        raise ValueError("split_role must be calibration or heldout")
    if task_selection not in {"all", "gsm8k", "math500"}:
        raise ValueError("unknown task selection")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num_shards")

    tasks = ("gsm8k", "math500") if task_selection == "all" else (task_selection,)
    all_items: list[Phase1WorkItem] = []
    for task in tasks:
        entries = manifest["splits"][task][split_role]
        if run_kind == "smoke":
            entries = entries[:1]
        for entry in entries:
            for rollout_index in _rollout_indices(run_kind):
                for policy_name in (
                    policies
                    if policies is not None
                    else _policy_names(task, split_role, run_kind)
                ):
                    all_items.append(
                        Phase1WorkItem(
                            task=task,
                            split_role=split_role,
                            run_kind=run_kind,
                            row_index=int(entry["row_index"]),
                            prompt_sha256=str(entry["prompt_sha256"]),
                            policy_name=policy_name,
                            rollout_index=rollout_index,
                        )
                    )
    return [
        item
        for index, item in enumerate(all_items)
        if index % num_shards == shard_index
    ]
