"""Safe filesystem discovery for deterministic dataset import workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileSelection:
    root: Path
    files: tuple[Path, ...]
    skipped: tuple[Path, ...]


def pick_files(root: str | Path, *, recursive: bool = True, extensions: tuple[str, ...] = (".txt", ".md", ".json", ".jsonl", ".pdf", ".parquet"), max_bytes: int = 16 * 1024 * 1024, exclude: tuple[str, ...] = (".git", ".venv")) -> FileSelection:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = [root] if root.is_file() else (root.rglob("*") if recursive else root.glob("*"))
    files, skipped = [], []
    allowed = {extension.lower() for extension in extensions}
    for path in sorted(path for path in candidates if path.is_file()):
        if any(part in exclude for part in path.parts) or path.suffix.lower() not in allowed or path.stat().st_size > max_bytes:
            skipped.append(path)
        else:
            files.append(path)
    return FileSelection(root, tuple(files), tuple(skipped))
