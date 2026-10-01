from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from dllm_order_transmission.artifacts import sha256_file
from dllm_order_transmission.condition_a_analyzer import analyze, audit_smoke
from dllm_order_transmission.condition_a_analyzer import build_freeze_payload
from dllm_order_transmission.condition_a_protocol import (
    ARTIFACT_SCHEMA_SHA256,
    EXPECTED_FORMAL_GENERATIONS,
    OPTION_B_FREEZE_SHA256,
    OPTION_B_REPAIR_AMENDMENT_SHA256,
    OPTION_B_REPORT_SHA256,
    build_work_items,
    validate_freeze,
)
from dllm_order_transmission.condition_a_runner import _policy_payload, _seed_key
from dllm_order_transmission.phase1_protocol import load_split_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "protocols" / "phase1_split_manifest_v1.json"
SCHEMA_PATH = ROOT / "protocols" / "condition_a_artifact_schema_v1.json"


def test_condition_a_cardinality_and_shards() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)
    all_items = build_work_items(manifest, run_kind="condition-a-formal")
    assert len(all_items) == EXPECTED_FORMAL_GENERATIONS
    assert len({item.key for item in all_items}) == EXPECTED_FORMAL_GENERATIONS

    shards = [
        build_work_items(
            manifest,
            run_kind="condition-a-formal",
            shard_index=index,
            num_shards=8,
        )
        for index in range(8)
    ]
    assert {len(shard) for shard in shards} == {1600}
    assert {item.key for shard in shards for item in shard} == {
        item.key for item in all_items
    }


def test_condition_a_smoke_is_one_item_per_task() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)
    items = build_work_items(manifest, run_kind="condition-a-smoke")
    assert len(items) == 2
    assert {item.task for item in items} == {"gsm8k", "math500"}
    assert {item.rollout_index for item in items} == {0}


def test_condition_a_uses_phase1_common_random_number_key() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)
    item = build_work_items(
        manifest, run_kind="condition-a-formal", task_selection="gsm8k"
    )[0]
    assert _seed_key(item).startswith("dllm-order-phase1-v1|gsm8k|20260823|")
    assert "ef_b32" not in _seed_key(item)


def test_condition_a_schema_hash_and_policy() -> None:
    assert sha256_file(SCHEMA_PATH) == ARTIFACT_SCHEMA_SHA256
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert _policy_payload() == {
        "name": "ef_b32",
        "steps": 256,
        "generation_length": 256,
        "block_length": 32,
        "temperature": 0.6,
        "cfg_scale": 0.0,
        "remasking": "high_entropy",
    }


def test_freeze_payload_records_matched_random_budget() -> None:
    trigger = {
        "p1_class": "entirely_below_zero",
        "p2_class": "resolved_negative",
        "run_condition_a": True,
    }
    payload = build_freeze_payload(
        analyzer_commit="a" * 40,
        runner_commit="a" * 40,
        trigger=trigger,
    )
    assert payload["cardinality"]["new_generations"] == 12_800
    assert payload["primary_endpoint"]["contrast"].startswith("CoverageAUC64")
    matched = payload["secondary_endpoints"]["matched_16_rollout_contrasts"]
    assert "16" in matched["metric"]
    assert "unmatched 64-rollout" in matched["reason"]
    assert payload["trigger"] == trigger


def test_validate_freeze_rejects_wrong_option_b_report(tmp_path: Path) -> None:
    payload = build_freeze_payload(
        analyzer_commit="a" * 40,
        runner_commit="a" * 40,
        trigger={
            "p1_class": "entirely_below_zero",
            "p2_class": "resolved_negative",
            "run_condition_a": True,
        },
    )
    payload["option_b_report_sha256"] = "0" * 64
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze mismatch"):
        validate_freeze(path)


def test_freeze_payload_binds_canonical_option_b_chain() -> None:
    payload = build_freeze_payload(
        analyzer_commit="a" * 40,
        runner_commit="a" * 40,
        trigger={"run_condition_a": True},
    )
    assert payload["option_b_freeze_sha256"] == OPTION_B_FREEZE_SHA256
    assert payload["option_b_report_sha256"] == OPTION_B_REPORT_SHA256
    assert (
        payload["option_b_repair_amendment_sha256"]
        == OPTION_B_REPAIR_AMENDMENT_SHA256
    )


def test_analyzer_handles_zero_cells_and_recovers_primary_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    condition_root = tmp_path / "condition"
    phase1_root = tmp_path / "phase1"
    manifest = {
        "splits": {
            task: {"heldout": [{"row_index": index}]}
            for index, task in enumerate(("gsm8k", "math500"))
        }
    }

    def write(path: Path, *, correct: bool, words: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "correct": correct,
                    "response": " ".join(["word"] * words),
                    "parser_status": "ok",
                }
            ),
            encoding="utf-8",
        )

    from dllm_order_transmission import condition_a_analyzer as module

    for index, task in enumerate(("gsm8k", "math500")):
        # MATH500 deliberately gives AO an all-zero cell.
        ef_successes = 8 if task == "gsm8k" else 2
        ao_successes = 4 if task == "gsm8k" else 0
        for rollout in range(64):
            write(
                module._condition_result_path(
                    condition_root, task, index, rollout
                ),
                correct=rollout < ef_successes,
                words=12,
            )
            write(
                module._phase1_result_path(
                    phase1_root, task, index, "ao_b32", rollout
                ),
                correct=rollout < ao_successes,
                words=10,
            )
        for rollout in range(16):
            write(
                module._phase1_result_path(
                    phase1_root, task, index, "random_b32", rollout
                ),
                correct=rollout < 1,
                words=11,
            )

    monkeypatch.setattr(
        module,
        "_validate_artifacts",
        lambda **kwargs: {
            "expected": 128,
            "present_and_valid": 128,
            "missing": 0,
            "invalid": 0,
            "outcome_fields_read": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_validate_markers",
        lambda root: {"markers": 8, "total_result_count": 128},
    )
    monkeypatch.setattr(module, "_validate_control_record", lambda **kwargs: None)
    report = analyze(
        condition_root=condition_root,
        phase1_root=phase1_root,
        manifest=manifest,
        freeze_payload={"runner_commit": "a" * 40},
        freeze_sha="b" * 64,
        validator=object(),
    )

    assert report["primary"]["pooled"]["point"] > 0
    assert report["primary"]["classification"] == "entirely_above_zero"
    assert report["length_guard"]["triggered"] is True
    assert report["summaries"]["math500"]["ao_b32"]["pass_at_1"] == 0


def test_smoke_audit_checks_entropy_order_and_trace_chain(tmp_path: Path) -> None:
    from dllm_order_transmission import condition_a_analyzer as module

    condition_root = tmp_path / "condition"
    freeze_sha = "b" * 64
    runner_commit = "a" * 40
    manifest = {
        "splits": {
            task: {
                "calibration": [
                    {"row_index": index, "prompt_sha256": str(index) * 64}
                ]
            }
            for index, task in enumerate(("gsm8k", "math500"), start=1)
        }
    }

    for index, task in enumerate(("gsm8k", "math500"), start=1):
        result_path = module._smoke_result_path(condition_root, task, index)
        trace_path = result_path.with_name(result_path.stem + ".trace.jsonl.gz")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(trace_path, mode="wt", encoding="utf-8") as handle:
            for step in range(256):
                block_start = (step // 32) * 32
                record = {
                    "global_step": step,
                    "active_block_start": block_start,
                    "active_block_end": block_start + 32,
                    "eligible_positions": [step],
                    "entropy": [1.0],
                    "top1_logits": [2.0],
                    "top2_logits": [1.0],
                    "logit_margins": [1.0],
                    "probability_margins": [0.5],
                    "candidate_probabilities": [0.6],
                    "eligible_entropy": [1.0],
                    "eligible_logit_margins": [1.0],
                    "selected_positions": [step],
                    "top64_token_ids": [list(range(64))],
                    "top64_logits": [[float(value) for value in range(64)]],
                    "full_logsumexp": [64.0],
                    "state_checksum_before": f"state-{step}",
                    "state_checksum_after": f"state-{step + 1}",
                }
                handle.write(json.dumps(record) + "\n")
        result_path.write_text(
            json.dumps(
                {
                    "condition_a_freeze_sha256": freeze_sha,
                    "split_manifest_sha256": (
                        "68e4343846e2a8c7c3f5fba12a2cca9cbc2866f316e4fdb49adaf7cd40243b9b"
                    ),
                    "run_kind": "condition-a-smoke",
                    "rollout_index": 0,
                    "seed_key": module._seed_key(task, index, 0),
                    "completion_status": "complete",
                    "runner_repository": {"commit": runner_commit, "dirty": False},
                    "source": {
                        "prompt_sha256": str(index) * 64,
                        "row_index": index,
                        "split_role": "calibration",
                    },
                    "trace_sha256": sha256_file(trace_path),
                    "environment_fingerprint": {
                        "trace_equivalence_verified_for_this_result": True
                    },
                }
            ),
            encoding="utf-8",
        )

    marker = (
        condition_root
        / "condition-a-v1/manifests/condition-a-smoke/all/shard-000-of-001.json"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "result_count": 2,
                "expected_keys_sha256": "c" * 64,
                "observed_keys_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    class Validator:
        @staticmethod
        def validate(record: dict) -> None:
            return None

    result = audit_smoke(
        condition_root=condition_root,
        manifest=manifest,
        freeze_sha=freeze_sha,
        runner_commit=runner_commit,
        validator=Validator(),
    )
    assert result["passed"] is True
    assert result["errors"] == []
