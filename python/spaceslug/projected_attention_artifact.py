"""Checksummed artifact for the projected-attention Spaceslug-Tiny reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import ByteTokenizer


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_projected_artifact(output: str | Path, model: ProjectedTinyAttentionModel, tokenizer: ByteTokenizer) -> dict:
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    (root / "tokenizer").mkdir(parents=True)
    (root / "tensors").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_bytes(_bytes({"id": tokenizer.identifier, "revision": tokenizer.revision, "fingerprint": tokenizer.fingerprint(), "vocab_size": tokenizer.vocab_size}))
    (root / "model.json").write_bytes(_bytes({"architecture": "spaceslug-tiny-projected-attention-cpu-reference", "vocab_size": model.vocab_size, "hidden_size": model.hidden_size, "use_positions": model.use_positions}))
    (root / "tensors" / "weights.json").write_bytes(_bytes({name: getattr(model, name) for name in ("embedding", "query", "key", "value", "output", "lm_head")}))
    files = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in sorted(path for path in root.rglob("*") if path.is_file())]
    revision = "sha256:" + hashlib.sha256(b"".join(_bytes(item) for item in files)).hexdigest()
    manifest = {"format": "spaceslug-model", "schema_version": 1, "model_id": "Spaceslug-Tiny", "architecture": "spaceslug-tiny-projected-attention-cpu-reference", "revision": revision, "tokenizer": {"id": tokenizer.identifier, "revision": tokenizer.revision, "fingerprint": tokenizer.fingerprint()}, "files": files}
    (root / "manifest.json").write_bytes(_bytes(manifest))
    return manifest
