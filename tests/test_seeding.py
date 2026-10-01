from dllm_order_transmission.seeding import derive_rollout_seed, rollout_seed_key


def test_seed_is_stable_and_policy_independent() -> None:
    assert rollout_seed_key(55, 0) == "dllm-order-phase0-v1|20260820|55|0"
    assert derive_rollout_seed(55, 0) == derive_rollout_seed(55, 0)
    assert derive_rollout_seed(55, 0) != derive_rollout_seed(55, 1)
    assert derive_rollout_seed(55, 0) != derive_rollout_seed(92, 0)


def test_seed_rejects_negative_indices() -> None:
    try:
        derive_rollout_seed(-1, 0)
    except ValueError as error:
        assert "query_index" in str(error)
    else:
        raise AssertionError("negative query index should fail")
