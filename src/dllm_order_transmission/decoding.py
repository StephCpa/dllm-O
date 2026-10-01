"""Output-preserving traced decoder based on public JustGRPO generation code."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .trace import StepTrace


TraceCallback = Callable[[StepTrace], None]


@dataclass(frozen=True)
class DecodeConfig:
    steps: int = 256
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.6
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336

    def validate(self) -> None:
        if self.steps <= 0 or self.gen_length <= 0 or self.block_length <= 0:
            raise ValueError("steps, gen_length, and block_length must be positive")
        if self.gen_length % self.block_length != 0:
            raise ValueError("gen_length must be divisible by block_length")
        num_blocks = self.gen_length // self.block_length
        if self.steps % num_blocks != 0:
            raise ValueError("steps must be divisible by the number of blocks")
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if self.remasking not in {"low_confidence", "random", "high_entropy"}:
            raise ValueError(
                "remasking must be low_confidence, random, or high_entropy"
            )


def _state_checksum(tokens: torch.Tensor) -> str:
    payload = tokens.detach().to(device="cpu", dtype=torch.int64).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _uniform_like(logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(
        logits.shape,
        dtype=torch.float64,
        device=logits.device,
        generator=generator,
    )


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Match the public LLaDA/JustGRPO categorical sampling transformation."""
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = _uniform_like(logits64, generator)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute each batch element's masked tokens over reverse steps."""
    if mask_index.ndim != 2:
        raise ValueError("mask_index must have shape [batch, positions]")
    if steps <= 0:
        raise ValueError("steps must be positive")
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    result = torch.zeros(
        (mask_index.shape[0], steps),
        device=mask_index.device,
        dtype=torch.int64,
    )
    result += base
    for batch_index in range(mask_index.shape[0]):
        result[batch_index, : int(remainder[batch_index].item())] += 1
    return result


def _extract_logits(model_output: Any) -> torch.Tensor:
    logits = getattr(model_output, "logits", model_output)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must be a tensor or expose a logits tensor")
    return logits


@torch.no_grad()
def generate_with_trace(
    model: Any,
    prompt: torch.Tensor,
    *,
    config: DecodeConfig,
    generator: torch.Generator | None = None,
    trace_callback: TraceCallback | None = None,
    full_distribution: bool = False,
) -> torch.Tensor:
    """Generate tokens while emitting pre-mutation trace events.

    Sampling and position selection follow JustGRPO commit
    1a2fddb5c6655597e63081c0af5ebb718a849f39. Trace calculations do not
    consume random numbers or mutate model state.
    """
    config.validate()
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape [batch, prompt_length]")
    if prompt.shape[0] > 1 and not torch.equal(
        prompt, prompt[0].unsqueeze(0).expand_as(prompt)
    ):
        raise ValueError("all prompts in a generation batch must be identical")
    if torch.any(prompt == config.mask_id):
        raise ValueError("prompt must not contain the mask token")

    batch_size, prompt_length = prompt.shape
    device = prompt.device
    x = torch.full(
        (batch_size, prompt_length + config.gen_length),
        config.mask_id,
        dtype=prompt.dtype,
        device=device,
    )
    x[:, :prompt_length] = prompt
    prompt_index = x != config.mask_id

    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks
    global_step = 0

    eligible_entropy = torch.full(
        x.shape, float("nan"), dtype=torch.float32, device=device
    )
    eligible_margin = torch.full_like(eligible_entropy, float("nan"))

    for block_index in range(num_blocks):
        block_start = prompt_length + block_index * config.block_length
        block_end = block_start + config.block_length
        block_mask_index = x[:, block_start:block_end] == config.mask_id
        transfer_counts = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_in_block in range(steps_per_block):
            checksum_before = _state_checksum(x) if trace_callback is not None else ""
            mask_index = x == config.mask_id

            if config.cfg_scale > 0:
                unconditional = x.clone()
                unconditional[prompt_index] = config.mask_id
                combined = torch.cat([x, unconditional], dim=0)
                conditional_logits = _extract_logits(model(combined))
                logits, unconditional_logits = torch.chunk(conditional_logits, 2, dim=0)
                logits = unconditional_logits + (config.cfg_scale + 1) * (
                    logits - unconditional_logits
                )
            else:
                logits = _extract_logits(model(x))

            noisy_logits = add_gumbel_noise(
                logits, config.temperature, generator=generator
            )
            candidate_ids = torch.argmax(noisy_logits, dim=-1)

            probabilities = F.softmax(logits.float(), dim=-1)
            entropy: torch.Tensor | None = None
            if config.remasking == "low_confidence":
                candidate_probabilities = torch.gather(
                    probabilities, dim=-1, index=candidate_ids.unsqueeze(-1)
                ).squeeze(-1)
                selection_scores = candidate_probabilities
            elif config.remasking == "random":
                candidate_probabilities = torch.rand(
                    candidate_ids.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                selection_scores = candidate_probabilities
            else:
                candidate_probabilities = torch.gather(
                    probabilities, dim=-1, index=candidate_ids.unsqueeze(-1)
                ).squeeze(-1)
                log_probabilities = F.log_softmax(logits.float(), dim=-1)
                entropy = -(probabilities * log_probabilities).sum(dim=-1)
                selection_scores = entropy

            candidate_ids = torch.where(mask_index, candidate_ids, x)
            eligible_mask = mask_index.clone()
            eligible_mask[:, :block_start] = False
            eligible_mask[:, block_end:] = False
            ranked_scores = torch.where(
                eligible_mask, selection_scores, -torch.inf
            )

            transfer_index = torch.zeros_like(candidate_ids, dtype=torch.bool)
            for batch_index in range(batch_size):
                count = int(transfer_counts[batch_index, step_in_block].item())
                if count <= 0:
                    continue
                if config.remasking == "high_entropy":
                    # Stable descending sort makes exact entropy ties resolve to
                    # the lower absolute position, as required by Condition A.
                    selected = torch.argsort(
                        ranked_scores[batch_index],
                        descending=True,
                        stable=True,
                    )[:count]
                else:
                    selected = torch.topk(
                        ranked_scores[batch_index], k=count
                    ).indices
                transfer_index[batch_index, selected] = True

            event_payloads: list[dict[str, Any]] = []
            if trace_callback is not None:
                if entropy is None:
                    log_probabilities = F.log_softmax(logits.float(), dim=-1)
                    entropy = -(probabilities * log_probabilities).sum(dim=-1)
                top_probabilities, top_probability_ids = torch.topk(
                    probabilities, k=2, dim=-1
                )
                top_logits, top_logit_ids = torch.topk(logits.float(), k=2, dim=-1)
                logit_margins = top_logits[..., 0] - top_logits[..., 1]
                probability_margins = (
                    top_probabilities[..., 0] - top_probabilities[..., 1]
                )

                active_mask = mask_index.clone()
                active_mask[:, :block_start] = False
                active_mask[:, block_end:] = False
                if step_in_block == 0:
                    eligible_entropy[active_mask] = entropy[active_mask]
                    eligible_margin[active_mask] = logit_margins[active_mask]

                if full_distribution:
                    top_m = min(64, logits.shape[-1])
                    top64_logits, top64_logit_ids = torch.topk(
                        logits.float(), k=top_m, dim=-1
                    )
                    full_logsumexp = torch.logsumexp(logits.float(), dim=-1)
                else:
                    top64_logits = top64_logit_ids = full_logsumexp = None

                for batch_index in range(batch_size):
                    positions = torch.nonzero(
                        active_mask[batch_index], as_tuple=False
                    ).squeeze(-1)
                    selected_positions = torch.nonzero(
                        transfer_index[batch_index], as_tuple=False
                    ).squeeze(-1)
                    event_payloads.append(
                        {
                            "batch_index": batch_index,
                            "positions": positions,
                            "selected_positions": selected_positions,
                            "entropy": entropy,
                            "top_probability_ids": top_probability_ids,
                            "top_logits": top_logits,
                            "top_logit_ids": top_logit_ids,
                            "logit_margins": logit_margins,
                            "probability_margins": probability_margins,
                            "top64_logits": top64_logits,
                            "top64_logit_ids": top64_logit_ids,
                            "full_logsumexp": full_logsumexp,
                        }
                    )

            x[transfer_index] = candidate_ids[transfer_index]

            if trace_callback is not None:
                checksum_after = _state_checksum(x)
                for payload in event_payloads:
                    batch_index = payload["batch_index"]
                    positions = payload["positions"]
                    selected_positions = payload["selected_positions"]

                    def values(tensor: torch.Tensor, index: torch.Tensor) -> tuple[Any, ...]:
                        return tuple(tensor[batch_index, index].detach().cpu().tolist())

                    event = StepTrace(
                        batch_index=batch_index,
                        global_step=global_step,
                        block_index=block_index,
                        step_in_block=step_in_block,
                        active_block_start=block_start,
                        active_block_end=block_end,
                        eligible_positions=tuple(positions.cpu().tolist()),
                        entropy=values(payload["entropy"], positions),
                        top1_token_ids=values(
                            payload["top_logit_ids"][..., 0], positions
                        ),
                        top2_token_ids=values(
                            payload["top_logit_ids"][..., 1], positions
                        ),
                        top1_logits=values(payload["top_logits"][..., 0], positions),
                        top2_logits=values(payload["top_logits"][..., 1], positions),
                        logit_margins=values(payload["logit_margins"], positions),
                        probability_margins=values(
                            payload["probability_margins"], positions
                        ),
                        candidate_token_ids=values(candidate_ids, positions),
                        candidate_probabilities=values(
                            candidate_probabilities, positions
                        ),
                        eligible_entropy=values(eligible_entropy, positions),
                        eligible_logit_margins=values(eligible_margin, positions),
                        selected_positions=tuple(selected_positions.cpu().tolist()),
                        selected_token_ids=values(candidate_ids, selected_positions),
                        state_checksum_before=checksum_before,
                        state_checksum_after=checksum_after,
                        top64_token_ids=(
                            tuple(
                                tuple(row)
                                for row in payload["top64_logit_ids"][
                                    batch_index, positions
                                ]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            if full_distribution
                            else None
                        ),
                        top64_logits=(
                            tuple(
                                tuple(row)
                                for row in payload["top64_logits"][
                                    batch_index, positions
                                ]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            if full_distribution
                            else None
                        ),
                        full_logsumexp=(
                            values(payload["full_logsumexp"], positions)
                            if full_distribution
                            else None
                        ),
                    )
                    trace_callback(event)

            global_step += 1

    return x
