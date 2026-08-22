"""Deterministic, inspectable Spaceslug Dataset Bundle (.dts) support."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    root: Path
    manifest: dict

    def records(self, split: str) -> list[dict]:
        path = self.root / "records" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def stats(self) -> dict[str, int]:
        return {split: len(self.records(split)) for split in ("train", "validation", "test")}


def create_bundle(
    output: str | Path,
    dataset_id: str,
    splits: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    tokenizer_id: str = "unassigned",
    tokenizer_revision: str = "unassigned",
    preprocessing_pipeline: str = "identity",
    preprocessing_revision: str = "spaceslug-main",
    seed: int = 0,
    sources: Iterable[str] = (),
    licenses: Iterable[str] = (),
) -> DatasetBundle:
    """Write a deterministic uncompressed .dts directory and return it."""
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    source_values = list(sources)
    license_values = list(licenses)
    root.mkdir(parents=True)
    (root / "records").mkdir()
    (root / "source").mkdir()
    (root / "preprocessing").mkdir()
    (root / "checksums").mkdir()

    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = sorted((dict(row) for row in splits.get(split, ())), key=lambda row: str(row["record_id"]))
        for row in rows:
            if not isinstance(row.get("record_id"), str) or not row["record_id"]:
                raise ValueError("every record requires a non-empty string record_id")
        path = root / "records" / f"{split}.jsonl"
        path.write_bytes(b"".join(_json_bytes(row) for row in rows))
        counts[split] = len(rows)

    (root / "source" / "sources.jsonl").write_bytes(b"".join(_json_bytes({"source": value}) for value in source_values))
    (root / "source" / "licenses.jsonl").write_bytes(b"".join(_json_bytes({"license": value}) for value in license_values))
    (root / "preprocessing" / "config.json").write_bytes(_json_bytes({
        "pipeline": preprocessing_pipeline, "revision": preprocessing_revision, "seed": seed
    }))

    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        files.append({"path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    revision_input = b"".join(_json_bytes(item) for item in files)
    revision = "sha256:" + hashlib.sha256(revision_input).hexdigest()
    manifest = {
        "format": "spaceslug-dataset", "schema_version": 1, "dataset_id": dataset_id,
        "revision": revision, "parent_revision": None, "record_count": sum(counts.values()),
        "splits": {split: counts[split] for split in ("train", "validation", "test")},
        "encoding": {"text": "utf-8", "compression": "none"},
        "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision, "vocab_size": 1},
        "preprocessing": {"pipeline": preprocessing_pipeline, "revision": preprocessing_revision, "seed": seed},
        "provenance": {"sources": source_values, "licenses": license_values, "privacy_status": "unknown"},
        "files": files,
    }
    (root / "manifest.json").write_bytes(_json_bytes(manifest))
    (root / "checksums" / "sha256.json").write_bytes(_json_bytes({item["path"]: item["sha256"] for item in files}))
    return DatasetBundle(root, manifest)


def verify_bundle(root: str | Path) -> DatasetBundle:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "spaceslug-dataset" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset bundle")
    expected_splits = {"train", "validation", "test"}
    if set(manifest.get("splits", {})) != expected_splits:
        raise ValueError("dataset must declare train, validation, and test splits")
    bundle = DatasetBundle(root, manifest)
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"checksum mismatch: {item['path']}")
    for split, expected_count in manifest["splits"].items():
        if len(bundle.records(split)) != expected_count:
            raise ValueError(f"record count mismatch for {split}")
    return bundle
