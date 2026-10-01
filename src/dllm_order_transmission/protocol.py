"""Constants frozen in the Phase 0 protocol."""

from __future__ import annotations

from dataclasses import dataclass

from .decoding import DecodeConfig


PROTOCOL_VERSION = "phase0-v1"
JUSTGRPO_COMMIT = "1a2fddb5c6655597e63081c0af5ebb718a849f39"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_DATASET_SIZE = 1319

QUERY_INDICES = (
    55, 92, 98, 110, 141, 147, 148, 311, 380, 396,
    430, 510, 555, 587, 598, 600, 635, 637, 645, 673,
    682, 792, 797, 826, 836, 848, 877, 880, 881, 883,
    894, 940, 982, 1023, 1056, 1070, 1072, 1081, 1117, 1132,
    1167, 1182, 1187, 1194, 1199, 1224, 1251, 1252, 1292, 1316,
)


@dataclass(frozen=True)
class Policy:
    name: str
    decode_config: DecodeConfig


POLICIES = (
    Policy(
        name="ar_b1",
        decode_config=DecodeConfig(
            steps=256,
            gen_length=256,
            block_length=1,
            temperature=0.6,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=126336,
        ),
    ),
    Policy(
        name="ao_b32",
        decode_config=DecodeConfig(
            steps=256,
            gen_length=256,
            block_length=32,
            temperature=0.6,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=126336,
        ),
    ),
)


def stage_query_indices(stage: str) -> tuple[int, ...]:
    if stage == "equivalence":
        return QUERY_INDICES[:5]
    if stage == "signal":
        return QUERY_INDICES
    raise ValueError(f"unknown stage: {stage}")


def stage_rollout_count(stage: str) -> int:
    if stage == "equivalence":
        return 4
    if stage == "signal":
        return 32
    raise ValueError(f"unknown stage: {stage}")
