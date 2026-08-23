"""Checksummed canonical artifact for the dense Spaceslug-Tiny CPU reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tiny_dense_model import TinyDenseCausalModel
from .tokenizer import ByteTokenizer


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dense_tiny_artifact(output: str | Path, model: TinyDenseCausalModel, tokenizer: ByteTokenizer) -> dict:
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    (root / "tokenizer").mkdir(parents=True)
    (root / "tensors").mkdir()
    tokenizer_data = {"id": tokenizer.identifier, "revision": tokenizer.revision,
                      "fingerprint": tokenizer.fingerprint(), "vocab_size": tokenizer.vocab_size,
                      "special_tokens": {"pad": tokenizer.pad_token, "bos": tokenizer.bos_token, "eos": tokenizer.eos_token}}
    model_data = {"architecture": "spaceslug-tiny-dense-causal-cpu-reference", "vocab_size": model.vocab_size,
                  "hidden_size": model.hidden_size}
    weights = {"embedding": model.embedding, "output": model.output, "output_bias": model.output_bias}
    (root / "tokenizer" / "tokenizer.json").write_bytes(_json_bytes(tokenizer_data))
    (root / "model.json").write_bytes(_json_bytes(model_data))
    (root / "tensors" / "weights.json").write_bytes(_json_bytes(weights))
    files = [{"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
             for path in sorted(path for path in root.rglob("*") if path.is_file())]
    revision = "sha256:" + hashlib.sha256(b"".join(_json_bytes(item) for item in files)).hexdigest()
    manifest = {"format": "spaceslug-model", "schema_version": 1, "model_id": "Spaceslug-Tiny",
                "revision": revision, "architecture": model_data["architecture"], "tokenizer": tokenizer_data,
                "precision": "float64-cpu-reference", "files": files}
    (root / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def load_dense_tiny_artifact(root: str | Path) -> tuple[TinyDenseCausalModel, ByteTokenizer, dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "spaceslug-model" or manifest.get("architecture") != "spaceslug-tiny-dense-causal-cpu-reference":
        raise ValueError("unsupported dense Tiny artifact")
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {item['path']}")
    token_data = json.loads((root / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = ByteTokenizer(identifier=token_data["id"], revision=token_data["revision"], vocab_size=token_data["vocab_size"])
    if tokenizer.fingerprint() != token_data["fingerprint"]:
        raise ValueError("tokenizer fingerprint mismatch")
    model_data = json.loads((root / "model.json").read_text(encoding="utf-8"))
    weights = json.loads((root / "tensors" / "weights.json").read_text(encoding="utf-8"))
    model = TinyDenseCausalModel(model_data["vocab_size"], model_data["hidden_size"], weights["embedding"], weights["output"], weights["output_bias"])
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabularies differ")
    return model, tokenizer, manifest
