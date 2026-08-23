"""Greedy CPU-reference inference for a loaded projected Tiny artifact."""

from __future__ import annotations

from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import ByteTokenizer


def next_token(model: ProjectedTinyAttentionModel, tokenizer: ByteTokenizer, text: str) -> int:
    """Return the deterministic argmax next token for a prompt."""
    tokens = tokenizer.encode(text)
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabularies differ")
    logits = model.logits_for_tokens(tokens)
    return max(range(len(logits)), key=logits.__getitem__)
