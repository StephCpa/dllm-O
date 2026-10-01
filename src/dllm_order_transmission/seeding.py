"""Stable seed derivation defined by the frozen Phase 0 protocol."""

from __future__ import annotations

import hashlib


PROTOCOL_PREFIX = "dllm-order-phase0-v1"
GLOBAL_SEED = 20260820


def rollout_seed_key(
    query_index: int,
    rollout_index: int,
    *,
    global_seed: int = GLOBAL_SEED,
    protocol_prefix: str = PROTOCOL_PREFIX,
) -> str:
    """Return the policy-independent common-random-number key."""
    if query_index < 0:
        raise ValueError("query_index must be nonnegative")
    if rollout_index < 0:
        raise ValueError("rollout_index must be nonnegative")
    return f"{protocol_prefix}|{global_seed}|{query_index}|{rollout_index}"


def derive_rollout_seed(
    query_index: int,
    rollout_index: int,
    *,
    global_seed: int = GLOBAL_SEED,
    protocol_prefix: str = PROTOCOL_PREFIX,
) -> int:
    """Derive a stable signed-64-bit-compatible seed from SHA-256."""
    key = rollout_seed_key(
        query_index,
        rollout_index,
        global_seed=global_seed,
        protocol_prefix=protocol_prefix,
    )
    raw = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, byteorder="big", signed=False) % (2**63 - 1)
