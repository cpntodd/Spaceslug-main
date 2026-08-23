"""Checksummed reader/writer for projected-attention Spaceslug-Tiny artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import ByteTokenizer


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> list[dict]:
    return [{"path": str(path.relative_to(root)), "sha256": _digest(path), "bytes": path.stat().st_size}
            for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")]


def write_projected_artifact(output: str | Path, model: ProjectedTinyAttentionModel, tokenizer: ByteTokenizer) -> dict:
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    (root / "tokenizer").mkdir(parents=True)
    (root / "tensors").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_bytes(_bytes({"id": tokenizer.identifier, "revision": tokenizer.revision, "fingerprint": tokenizer.fingerprint(), "vocab_size": tokenizer.vocab_size}))
    (root / "model.json").write_bytes(_bytes({"architecture": "spaceslug-tiny-projected-attention-cpu-reference", "vocab_size": model.vocab_size, "hidden_size": model.hidden_size, "use_positions": model.use_positions}))
    (root / "tensors" / "weights.json").write_bytes(_bytes({name: getattr(model, name) for name in ("embedding", "query", "key", "value", "output", "lm_head")}))
    files = _files(root)
    revision = "sha256:" + hashlib.sha256(b"".join(_bytes(item) for item in files)).hexdigest()
    manifest = {"format": "spaceslug-model", "schema_version": 1, "model_id": "Spaceslug-Tiny", "architecture": "spaceslug-tiny-projected-attention-cpu-reference", "revision": revision, "tokenizer": {"id": tokenizer.identifier, "revision": tokenizer.revision, "fingerprint": tokenizer.fingerprint()}, "files": files}
    (root / "manifest.json").write_bytes(_bytes(manifest))
    return manifest


def load_projected_artifact(root: str | Path) -> tuple[ProjectedTinyAttentionModel, ByteTokenizer, dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "spaceslug-model" or manifest.get("schema_version") != 1 or manifest.get("architecture") != "spaceslug-tiny-projected-attention-cpu-reference":
        raise ValueError("unsupported projected Tiny artifact")
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _digest(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {item['path']}")
    token_data = json.loads((root / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = ByteTokenizer(identifier=token_data["id"], revision=token_data["revision"], vocab_size=token_data["vocab_size"])
    if tokenizer.fingerprint() != token_data["fingerprint"]:
        raise ValueError("tokenizer fingerprint mismatch")
    model_data = json.loads((root / "model.json").read_text(encoding="utf-8"))
    weights = json.loads((root / "tensors" / "weights.json").read_text(encoding="utf-8"))
    model = ProjectedTinyAttentionModel(model_data["vocab_size"], model_data["hidden_size"], model_data.get("use_positions", True))
    for name in ("embedding", "query", "key", "value", "output", "lm_head"):
        setattr(model, name, weights[name])
    return model, tokenizer, manifest
