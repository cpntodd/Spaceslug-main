"""Deterministic padded causal batches with explicit target-only loss masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class CausalBatch:
    input_tokens: list[list[int]]
    target_tokens: list[list[int]]
    loss_mask: list[list[bool]]

    @property
    def token_count(self) -> int:
        return sum(sum(row) for row in self.loss_mask)


def target_only_batches(records: Iterable[dict], tokenizer: ByteTokenizer, batch_size: int) -> list[CausalBatch]:
    """Create fixed-order padded teacher-forced batches from prompt/target records.

    `loss_mask[i]` applies to prediction of `target_tokens[i]`; prompt-token
    predictions and PAD targets are always excluded from the reduction.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    examples: list[tuple[list[int], list[int], list[bool]]] = []
    for record in records:
        prompt, target = record.get("prompt"), record.get("target")
        if not isinstance(prompt, str) or not isinstance(target, str) or not target:
            raise ValueError("target-only records require non-empty string prompt and target")
        prefix = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        completion = tokenizer.encode(target, add_bos=False, add_eos=True)
        sequence = prefix + completion
        examples.append((sequence[:-1], sequence[1:], [False] * (len(prefix) - 1) + [True] * len(completion)))
    if not examples:
        raise ValueError("no target-only records")
    batches: list[CausalBatch] = []
    for start in range(0, len(examples), batch_size):
        group = examples[start:start + batch_size]
        width = max(len(inputs) for inputs, _, _ in group)
        inputs, targets, masks = [], [], []
        for source, target, mask in group:
            padding = width - len(source)
            inputs.append(source + [tokenizer.pad_token] * padding)
            targets.append(target + [tokenizer.pad_token] * padding)
            masks.append(mask + [False] * padding)
        batches.append(CausalBatch(inputs, targets, masks))
    return batches
