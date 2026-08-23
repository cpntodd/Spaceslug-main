"""Inspectable, checksummed artifact for the Spaceslug-Tiny CPU reference model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tiny_model import TinyBigramModel
from .tokenizer import ByteTokenizer


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tiny_artifact(output: str | Path, model: TinyBigramModel, tokenizer: ByteTokenizer) -> dict:
    """Create a new canonical Tiny artifact directory; never overwrite one."""
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    (root / "tokenizer").mkdir(parents=True)
    (root / "tensors").mkdir()
    tokenizer_payload = {
        "id": tokenizer.identifier,
        "revision": tokenizer.revision,
        "fingerprint": tokenizer.fingerprint(),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {"pad": tokenizer.pad_token, "bos": tokenizer.bos_token, "eos": tokenizer.eos_token},
    }
    model_payload = {"architecture": "spaceslug-tiny-bigram-cpu-reference", "vocab_size": model.vocab_size}
    weights_payload = {"vocab_size": model.vocab_size, "weights": model.weights}
    (root / "tokenizer" / "tokenizer.json").write_bytes(_json_bytes(tokenizer_payload))
    (root / "model.json").write_bytes(_json_bytes(model_payload))
    (root / "tensors" / "weights.json").write_bytes(_json_bytes(weights_payload))
    files = [{"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
             for path in sorted(path for path in root.rglob("*") if path.is_file())]
    revision = "sha256:" + hashlib.sha256(b"".join(_json_bytes(item) for item in files)).hexdigest()
    manifest = {
        "format": "spaceslug-model", "schema_version": 1, "model_id": "Spaceslug-Tiny",
        "revision": revision, "architecture": model_payload["architecture"],
        "tokenizer": tokenizer_payload, "precision": "float64-cpu-reference", "files": files,
    }
    (root / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def load_tiny_artifact(root: str | Path) -> tuple[TinyBigramModel, ByteTokenizer, dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "spaceslug-model" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported Spaceslug model artifact")
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"artifact checksum mismatch: {item['path']}")
    token_data = json.loads((root / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = ByteTokenizer(identifier=token_data["id"], revision=token_data["revision"], vocab_size=token_data["vocab_size"])
    if tokenizer.fingerprint() != token_data["fingerprint"]:
        raise ValueError("tokenizer fingerprint mismatch")
    payload = json.loads((root / "tensors" / "weights.json").read_text(encoding="utf-8"))
    model = TinyBigramModel(int(payload["vocab_size"]), payload["weights"])
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabularies differ")
    return model, tokenizer, manifest
