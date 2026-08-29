"""Stage 1 curated Tiny-agent dataset preparation and hygiene.

This module is deterministic and dependency-free.  It keeps rich records while
emitting the prompt/target fields consumed by the existing Tiny trainers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping

from .dataset import DatasetBundle, create_bundle
from .workspace import DEFAULT_MAX_BYTES, extract_text, infer_kind

SCHEMA_VERSION = 1
FORMATS = frozenset({"instruction", "coding", "troubleshooting", "definition", "conversation", "tool_trace", "evaluation"})
_SPLITS = ("train", "validation", "test")
_SECRET = re.compile(r"(?i)(?:api[_ -]?key|token|password|secret)\s*[:=]\s*\S+")
_WS = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize whitespace without changing code-block contents semantically."""
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(_WS.sub(" ", line).strip() for line in value.split("\n")).strip()


def _content(record: Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(record.get("messages"), list):
        messages = [m for m in record["messages"] if isinstance(m, Mapping)]
        prompt = "\n".join(f"{m.get('role', 'user')}: {normalize_text(str(m.get('content', '')))}" for m in messages[:-1])
        target = normalize_text(str(messages[-1].get("content", ""))) if messages else ""
        return prompt, target
    return normalize_text(str(record.get("prompt", record.get("instruction", record.get("text", ""))))), normalize_text(str(record.get("target", record.get("output", record.get("response", "")))))


def validate_record(record: Mapping[str, Any], *, evaluation: bool = False) -> dict[str, Any]:
    """Return a canonical record or raise for unusable training data."""
    if not isinstance(record, Mapping):
        raise ValueError("record must be an object")
    if any(isinstance(value, float) and not math.isfinite(value) for value in record.values()):
        raise ValueError("record contains non-finite numeric metadata")
    rid_value = record.get("record_id", "")
    if not isinstance(rid_value, str):
        raise ValueError("record_id must be a string")
    rid = rid_value.strip()
    fmt = str(record.get("format", "instruction")).strip().lower()
    if not rid:
        raise ValueError("record_id is required")
    if fmt not in FORMATS:
        raise ValueError(f"unsupported record format: {fmt}")
    prompt, target = _content(record)
    if not prompt or not target:
        raise ValueError(f"record {rid} requires non-empty prompt and target")
    if len(prompt) > 16000 or len(target) > 16000:
        raise ValueError(f"record {rid} exceeds 16000-character limit")
    result = dict(record)
    result.update({"schema_version": SCHEMA_VERSION, "record_id": rid, "format": fmt, "prompt": prompt, "target": target})
    result["topic"] = str(record.get("topic", "general")).strip() or "general"
    result["group_id"] = str(record.get("group_id", record.get("source", rid))).strip() or rid
    result["quality_status"] = str(record.get("quality_status", "approved" if evaluation else "review")).strip()
    result["hygiene_flags"] = quality_flags(result)
    if "source" in record:
        result["source"] = str(record["source"])
    return result


def quality_flags(record: Mapping[str, Any]) -> list[str]:
    prompt, target = _content(record)
    flags: list[str] = []
    if _SECRET.search(prompt) or _SECRET.search(target):
        flags.append("possible-secret")
    if len(target) < 8:
        flags.append("very-short-answer")
    words = target.split()
    if words and len(words) >= 8 and len(set(words)) <= max(2, len(words) // 4):
        flags.append("repetitive-answer")
    if "TODO" in target or "[insert" in target.lower():
        flags.append("unresolved-placeholder")
    return flags


def normalized_key(record: Mapping[str, Any]) -> str:
    prompt, target = _content(record)
    return hashlib.sha256((normalize_text(prompt).casefold() + "\n" + normalize_text(target).casefold()).encode()).hexdigest()


def deduplicate(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove exact normalized duplicates, retaining the first deterministically."""
    seen: set[str] = set(); ids: set[str] = set(); output: list[dict[str, Any]] = []; duplicates: list[str] = []
    for raw in records:
        record = validate_record(raw)
        if record["record_id"] in ids:
            raise ValueError(f"duplicate record_id: {record['record_id']}")
        ids.add(record["record_id"])
        key = normalized_key(record)
        if key in seen:
            duplicates.append(record["record_id"])
            continue
        seen.add(key); output.append(record)
    return output, duplicates


def grouped_split(records: Iterable[Mapping[str, Any]], *, seed: int = 0, ratios: tuple[float, float, float] = (.8, .1, .1)) -> dict[str, list[dict[str, Any]]]:
    """Assign whole groups to reproducible train/validation/test splits."""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative and sum to one")
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw in records:
        record = validate_record(raw)
        groups.setdefault(str(record["group_id"]), []).append(record)
    keys = sorted(groups); Random(seed).shuffle(keys)
    total = sum(len(v) for v in groups.values()); targets = [total * r for r in ratios]
    result = {name: [] for name in _SPLITS}; counts = [0, 0, 0]
    for key in keys:
        index = min(range(3), key=lambda i: (counts[i] - targets[i], counts[i]))
        result[_SPLITS[index]].extend(groups[key]); counts[index] += len(groups[key])
    for values in result.values(): values.sort(key=lambda r: r["record_id"])
    return result


def build_report(records: Iterable[Mapping[str, Any]], splits: Mapping[str, Iterable[Mapping[str, Any]]] | None = None, duplicates: Iterable[str] = ()) -> dict[str, Any]:
    rows = [validate_record(r) for r in records]
    duplicate_ids = list(duplicates)
    flag_counts = Counter(flag for row in rows for flag in row["hygiene_flags"])
    return {"schema_version": SCHEMA_VERSION, "records": len(rows), "formats": dict(sorted(Counter(r["format"] for r in rows).items())), "topics": dict(sorted(Counter(r["topic"] for r in rows).items())), "groups": len({r["group_id"] for r in rows}), "flagged": sum(bool(r["hygiene_flags"]) for r in rows), "flags": dict(sorted(flag_counts.items())), "duplicates_removed": len(duplicate_ids), "duplicate_ids": duplicate_ids, "splits": {name: len(list((splits or {}).get(name, ()))) for name in _SPLITS}}


def build_stage1_bundle(output: str | Path, dataset_id: str, records: Iterable[Mapping[str, Any]], *, seed: int = 0, sources: Iterable[str] = (), licenses: Iterable[str] = ()) -> tuple[DatasetBundle, dict[str, Any]]:
    unique, duplicates = deduplicate(records)
    evaluation = [r for r in unique if r.get("format") == "evaluation" or r.get("quality_status") == "evaluation"]
    trainable = [r for r in unique if r not in evaluation and r.get("quality_status") != "rejected" and "possible-secret" not in r.get("hygiene_flags", [])]
    splits = grouped_split(trainable, seed=seed)
    # Evaluation records are immutable and never enter training; retain them in test.
    splits["test"].extend(sorted(evaluation, key=lambda r: r["record_id"]))
    bundle = create_bundle(output, dataset_id, splits, tokenizer_id="spaceslug-byte", tokenizer_revision="v1", preprocessing_pipeline="spaceslug-stage1", preprocessing_revision=f"schema-{SCHEMA_VERSION}", seed=seed, sources=sources, licenses=licenses)
    report = build_report(unique, splits, duplicates)
    (Path(output) / "stage1-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return bundle, report


def import_folder(folder: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES, extensions: tuple[str, ...] = (".txt", ".md", ".json", ".jsonl", ".pdf")) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recursively import supported documents as reviewable Stage 1 records."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"folder does not exist: {root}")
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {"folder": str(root), "files_seen": 0, "files_imported": 0, "files_skipped": [], "files_failed": []}
    allowed = {suffix.lower() for suffix in extensions}
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(root))):
        report["files_seen"] += 1
        if path.suffix.lower() not in allowed:
            report["files_skipped"].append({"path": str(path.relative_to(root)), "reason": "unsupported-extension"})
            continue
        if path.stat().st_size > max_bytes:
            report["files_skipped"].append({"path": str(path.relative_to(root)), "reason": "too-large"})
            continue
        try:
            data = path.read_bytes()
            source = str(path)
            extracted = extract_text(data, infer_kind(path.name), source_id=source)
            for index, raw in enumerate(extracted):
                text = normalize_text(str(raw.get("text", "")))
                if not text:
                    continue
                # Keep every source byte represented while producing records that
                # fit the existing Tiny context/training contract.
                chunk_size = 12000
                chunks = [text[offset:offset + chunk_size] for offset in range(0, len(text), chunk_size)]
                for chunk_index, chunk in enumerate(chunks):
                    records.append({"record_id": hashlib.sha256(f"{source}\n{index}\n{chunk_index}".encode()).hexdigest(), "format": "instruction", "prompt": f"Explain the following material from {path.name}, section {chunk_index + 1}.", "target": chunk, "topic": path.parent.name or "general", "group_id": source, "source": source, "quality_status": "review"})
            report["files_imported"] += 1
        except Exception as exc:
            report["files_failed"].append({"path": str(path.relative_to(root)), "reason": f"{type(exc).__name__}: {exc}"})
    report["records_extracted"] = len(records)
    return records, report


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try: result.append(json.loads(line))
                except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON on line {line_number}") from exc
    return result


def prepare_discord_messages(messages: Iterable[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    """Convert Discord-style exports into reviewable conversation records."""
    rows = []
    for index, message in enumerate(messages):
        content = normalize_text(str(message.get("content", "")))
        if not content or message.get("author", {}).get("bot", False): continue
        author = message.get("author", {})
        author_name = author.get("name", author.get("username", "user")) if isinstance(author, Mapping) else "user"
        author_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(author_name))
        rows.append({"record_id": f"{source}-{index:06d}", "format": "conversation", "messages": [{"role": "user", "content": f"{author_name}: {content}"}], "source": source, "group_id": str(message.get("thread", source)), "quality_status": "review"})
    return rows

__all__ = ["SCHEMA_VERSION", "build_report", "build_stage1_bundle", "deduplicate", "grouped_split", "import_folder", "load_jsonl", "normalize_text", "prepare_discord_messages", "quality_flags", "validate_record"]
