"""Deterministic byte tokenizer used by the Spaceslug-Tiny reference path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ByteTokenizer:
    """Versioned UTF-8 byte tokenizer with explicit BOS/EOS/PAD tokens."""

    identifier: str = "spaceslug-byte"
    revision: str = "v1"
    pad_token: int = 0
    bos_token: int = 1
    eos_token: int = 2
    byte_offset: int = 3
    vocab_size: int = 259

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_token)
        tokens.extend(byte + self.byte_offset for byte in text.encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: list[int], *, skip_special: bool = True) -> str:
        values: list[int] = []
        for token in tokens:
            if self.byte_offset <= token < self.vocab_size:
                values.append(token - self.byte_offset)
            elif not skip_special:
                raise ValueError(f"cannot decode special token: {token}")
        return bytes(values).decode("utf-8", errors="strict")

    def fingerprint(self) -> str:
        specification = f"{self.identifier}:{self.revision}:{self.pad_token}:{self.bos_token}:{self.eos_token}:{self.byte_offset}:{self.vocab_size}"
        return "sha256:" + hashlib.sha256(specification.encode("ascii")).hexdigest()


def default_tokenizer() -> ByteTokenizer:
    return ByteTokenizer()
