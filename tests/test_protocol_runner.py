from __future__ import annotations

import pytest

from dllm_order_transmission.protocol import (
    EXPECTED_DATASET_SIZE,
    POLICIES,
    QUERY_INDICES,
    stage_query_indices,
    stage_rollout_count,
)
from dllm_order_transmission.runner import build_work_items, parse_args


def test_frozen_query_set_is_valid() -> None:
    assert len(QUERY_INDICES) == 50
    assert len(set(QUERY_INDICES)) == 50
    assert tuple(sorted(QUERY_INDICES)) == QUERY_INDICES
    assert min(QUERY_INDICES) >= 0
    assert max(QUERY_INDICES) < EXPECTED_DATASET_SIZE


def test_frozen_policy_semantics() -> None:
    assert [policy.name for policy in POLICIES] == ["ar_b1", "ao_b32"]
    assert [policy.decode_config.block_length for policy in POLICIES] == [1, 32]
    assert all(policy.decode_config.temperature == 0.6 for policy in POLICIES)


def test_stage_sizes() -> None:
    assert len(stage_query_indices("equivalence")) == 5
    assert stage_rollout_count("equivalence") == 4
    assert len(build_work_items("equivalence", 0, 1)) == 20
    assert len(stage_query_indices("signal")) == 50
    assert stage_rollout_count("signal") == 32
    assert len(build_work_items("signal", 0, 1)) == 1600


def test_shards_are_disjoint_and_complete() -> None:
    shards = [set(build_work_items("signal", index, 3)) for index in range(3)]
    assert not (shards[0] & shards[1])
    assert not (shards[0] & shards[2])
    assert not (shards[1] & shards[2])
    assert len(set.union(*shards)) == 1600


def test_invalid_shard_fails() -> None:
    with pytest.raises(ValueError):
        build_work_items("signal", 3, 3)


def test_cli_requires_frozen_inputs(tmp_path) -> None:
    args = parse_args(
        [
            "--stage",
            "equivalence",
            "--justgrpo-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--model-path",
            str(tmp_path / "model"),
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--dry-run",
        ]
    )
    assert args.stage == "equivalence"
    assert args.dry_run is True
    assert args.model_path == tmp_path / "model"
    assert args.dataset_path == tmp_path / "dataset"
