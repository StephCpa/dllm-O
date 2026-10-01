from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class ToyOutput:
    logits: torch.Tensor


class ToyMaskedModel:
    """Small deterministic bidirectional model for decoder invariant tests."""

    def __init__(self, vocab_size: int = 8, mask_id: int = 7) -> None:
        self.vocab_size = vocab_size
        self.mask_id = mask_id

    def __call__(self, tokens: torch.Tensor) -> ToyOutput:
        batch_size, length = tokens.shape
        device = tokens.device
        vocab = torch.arange(self.vocab_size, device=device).view(1, 1, -1)
        positions = torch.arange(length, device=device).view(1, -1)
        observed = torch.where(tokens == self.mask_id, 0, tokens).sum(dim=1, keepdim=True)
        targets = (positions + observed) % (self.vocab_size - 1)
        distance = torch.abs(vocab - targets.unsqueeze(-1)).float()
        logits = -0.12 * distance
        logits[..., self.mask_id] = -20.0
        return ToyOutput(logits=logits.expand(batch_size, -1, -1).contiguous())
