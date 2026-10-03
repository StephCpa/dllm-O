from __future__ import annotations

import math

import pytest
import torch

from conftest import ToyMaskedModel
from dllm_order_transmission.decoding import (
    DecodeConfig,
    generate_with_trace,
    get_num_transfer_tokens,
)
from dllm_order_transmission.trace import (
    ListTraceSink,
    reconstruct_completion,
    validate_trace,
)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_transfer_counts_preserve_total() -> None:
    mask = torch.ones((2, 7), dtype=torch.bool)
    counts = get_num_transfer_tokens(mask, steps=4)
    assert counts.tolist() == [[2, 2, 2, 1], [2, 2, 2, 1]]
    assert torch.equal(counts.sum(dim=1), torch.tensor([7, 7]))


@pytest.mark.parametrize("block_length", [1, 32, 256])
def test_protocol_block_schedules_satisfy_trace_invariants(block_length: int) -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=256,
        gen_length=256,
        block_length=block_length,
        temperature=0.0,
        mask_id=7,
    )
    sink = ListTraceSink()
    output = generate_with_trace(model, prompt, config=config, trace_callback=sink)

    errors = validate_trace(
        sink.events,
        prompt_length=prompt.shape[1],
        gen_length=config.gen_length,
        total_steps=config.steps,
        block_length=config.block_length,
    )
    assert errors == []
    reconstructed = reconstruct_completion(
        sink.events,
        prompt_length=prompt.shape[1],
        gen_length=config.gen_length,
    )
    assert reconstructed == output[0, prompt.shape[1] :].tolist()


def test_strict_ar_closure_is_zero_by_construction() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=1,
        temperature=0.0,
        mask_id=7,
    )
    sink = ListTraceSink()
    generate_with_trace(model, prompt, config=config, trace_callback=sink)
    for event in sink.events:
        selected = event.selected_positions[0]
        index = event.eligible_positions.index(selected)
        assert event.entropy[index] == event.eligible_entropy[index]
        assert event.logit_margins[index] == event.eligible_logit_margins[index]


@pytest.mark.parametrize("block_length", [1, 4, 8])
def test_trace_callback_does_not_change_stochastic_output(block_length: int) -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=block_length,
        temperature=0.6,
        mask_id=7,
    )
    untraced = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(1234),
    )
    sink = ListTraceSink()
    traced = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(1234),
        trace_callback=sink,
    )
    assert torch.equal(untraced, traced)


def test_different_rollout_seeds_change_stochastic_generation() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.6,
        mask_id=7,
    )
    first = generate_with_trace(
        model, prompt, config=config, generator=make_generator(10)
    )
    second = generate_with_trace(
        model, prompt, config=config, generator=make_generator(11)
    )
    assert not torch.equal(first, second)


def test_full_distribution_trace_fields() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=256,
        gen_length=256,
        block_length=32,
        temperature=0.0,
        mask_id=7,
    )
    sink = ListTraceSink()
    generate_with_trace(
        model,
        prompt,
        config=config,
        trace_callback=sink,
        full_distribution=True,
    )
    assert len(sink.events) == config.steps
    for event in sink.events:
        payload = event.to_dict()
        assert "top64_token_ids" in payload
        assert "top64_logits" in payload
        assert "full_logsumexp" in payload
        assert len(payload["top64_token_ids"]) == len(event.eligible_positions)
        for row in payload["top64_token_ids"]:
            assert len(row) == min(64, model.vocab_size)
        for row in payload["top64_logits"]:
            assert len(row) == min(64, model.vocab_size)
            assert all(math.isfinite(value) for value in row)
        assert all(
            math.isfinite(value) for value in payload["full_logsumexp"]
        )
    # Plain Phase 1-style traces must not carry the new fields.
    plain_sink = ListTraceSink()
    generate_with_trace(
        model, prompt, config=config, trace_callback=plain_sink
    )
    for event in plain_sink.events:
        payload = event.to_dict()
        assert "top64_token_ids" not in payload
        assert "full_logsumexp" not in payload


def test_full_distribution_leaves_output_unchanged() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=256,
        gen_length=256,
        block_length=16,
        temperature=0.0,
        mask_id=7,
    )
    plain = generate_with_trace(
        model, prompt, config=config, generator=make_generator(7)
    )
    with_fields = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(7),
        trace_callback=ListTraceSink(),
        full_distribution=True,
    )
    assert torch.equal(plain, with_fields)


def test_high_entropy_selects_most_uncertain_eligible_position() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.0,
        remasking="high_entropy",
        mask_id=7,
    )
    sink = ListTraceSink()
    generate_with_trace(model, prompt, config=config, trace_callback=sink)

    for event in sink.events:
        best_entropy = max(event.entropy)
        expected = min(
            position
            for position, entropy in zip(
                event.eligible_positions, event.entropy, strict=True
            )
            if entropy == best_entropy
        )
        assert event.selected_positions == (expected,)


def test_high_entropy_trace_does_not_change_output() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.6,
        remasking="high_entropy",
        mask_id=7,
    )
    plain = generate_with_trace(
        model, prompt, config=config, generator=make_generator(20260906)
    )
    traced = generate_with_trace(
        model,
        prompt,
        config=config,
        generator=make_generator(20260906),
        trace_callback=ListTraceSink(),
        full_distribution=True,
    )
    assert torch.equal(plain, traced)


class SharedBinaryModel:
    """Every position has the same distribution: P(token 0)=0.9, P(token 1)=0.1."""

    vocab_size = 8
    mask_id = 7

    def __call__(self, tokens: torch.Tensor):
        batch_size, length = tokens.shape
        logits = torch.full((batch_size, length, self.vocab_size), -30.0)
        logits[..., 0] = math.log(0.9)
        logits[..., 1] = math.log(0.1)
        from conftest import ToyOutput

        return ToyOutput(logits=logits)


def first_step_committed_tokens(remasking: str, batch_size: int = 600) -> list[int]:
    config = DecodeConfig(
        steps=8, gen_length=8, block_length=8, temperature=1.0, remasking=remasking, mask_id=7
    )
    prompt = torch.tensor([[2, 3]], dtype=torch.long).expand(batch_size, -1).contiguous()
    sink = ListTraceSink()
    generate_with_trace(
        SharedBinaryModel(), prompt, config=config, generator=make_generator(20261003), trace_callback=sink
    )
    first = [event for event in sink.events if event.global_step == 0]
    return [event.selected_token_ids[0] for event in first]


def test_cf_filters_candidates_while_resample_restores_the_tempered_distribution() -> None:
    # With identical position distributions, CF commits token 0 unless all
    # eight candidates are token 1 (probability 1 - 0.1**8). The resample
    # control keeps the same position rule but commits a fresh draw, so token 0
    # appears with its model probability, 0.9.
    cf = first_step_committed_tokens("low_confidence")
    resample = first_step_committed_tokens("low_confidence_resample")
    assert sum(token == 0 for token in cf) / len(cf) > 0.995
    share = sum(token == 0 for token in resample) / len(resample)
    assert 0.85 < share < 0.95


def test_cf_resample_selects_positions_by_the_cf_rule() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    config = DecodeConfig(
        steps=8, gen_length=8, block_length=8, temperature=0.6,
        remasking="low_confidence_resample", mask_id=7,
    )
    sink = ListTraceSink()
    output = generate_with_trace(
        model, prompt, config=config, generator=make_generator(7), trace_callback=sink
    )
    for event in sink.events:
        best = max(event.candidate_probabilities)
        (selected,) = event.selected_positions
        index = event.eligible_positions.index(selected)
        assert event.candidate_probabilities[index] == best
    reconstructed = reconstruct_completion(sink.events, prompt_length=2, gen_length=8)
    assert torch.equal(torch.tensor(reconstructed), output[0, 2:])


def test_cf_resample_equals_cf_at_zero_temperature() -> None:
    model = ToyMaskedModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    outputs = []
    for remasking in ("low_confidence", "low_confidence_resample"):
        config = DecodeConfig(
            steps=32, gen_length=32, block_length=32, temperature=0.0, remasking=remasking, mask_id=7
        )
        outputs.append(generate_with_trace(model, prompt, config=config))
    assert torch.equal(outputs[0], outputs[1])
