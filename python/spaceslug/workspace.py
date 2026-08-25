"""Dependency-free Phase 1 dataset ingestion and workspace service.

This module is the headless ingestion/workspace service for Spaceslug-main. It
has no GUI dependency, no third-party packages, and performs HTTP(S) access only
after explicit user approval. It stages content-addressed sources, records
SHA-256/retrieval provenance and license confirmation, extracts basic text, and
builds deterministic ``.dts`` bundles through
:func:`spaceslug.dataset.create_bundle`.

The SearXNG-compatible query helper never fetches a search result document on
its own; a selected result URL must be passed back through
:meth:`WorkspaceService.import_url` with ``approved=True``.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Mapping

from .dataset import DatasetBundle, create_bundle

# --- Limits and defaults ----------------------------------------------------

DEFAULT_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
ALLOWED_LOCAL_EXTENSIONS = (".txt", ".md", ".jsonl")
_READ_CHUNK = 64 * 1024
_USER_AGENT = "spaceslug-ingest/0.1"

# Text fields recognized by the basic JSON/JSONL extractor, in priority order.
_TEXT_FIELDS = ("text", "content", "body", "prompt", "completion", "instruction", "response")

# Block-level HTML elements that introduce a line break in the basic extractor.
_BLOCK_TAGS = {
    "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "head", "li", "p", "pre", "section", "table", "td", "th", "title", "tr",
    "ul", "ol",
}
_IGNORED_TAGS = {"script", "style", "noscript"}


# --- Errors ------------------------------------------------------------------

class IngestionError(Exception):
    """Base class for ingestion and workspace failures."""


class UnsupportedKindError(IngestionError):
    """A source kind or extension is not supported."""


class ContentTooLargeError(IngestionError):
    """A source exceeded the configured byte limit."""


class FetchNotApprovedError(IngestionError):
    """An HTTP(S) fetch was requested without explicit user approval."""


class FetchTimeoutError(IngestionError):
    """A fetch exceeded the configured time limit."""


class InvalidURLError(IngestionError):
    """A URL failed validation."""


class LicenseRequiredError(IngestionError):
    """A license confirmation is required before a source can be ingested."""


# --- Small helpers -----------------------------------------------------------

def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8").lstrip("\ufeff")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"source is not valid UTF-8 text: {exc}") from exc


# --- URL validation ----------------------------------------------------------

def validate_http_url(url: str) -> str:
    """Validate an absolute HTTP(S) URL and return it unchanged.

    Raises :class:`InvalidURLError` for non-strings, non-HTTP(S) schemes,
    missing hosts, and URLs with embedded credentials.
    """
    if not isinstance(url, str):
        raise InvalidURLError(f"URL must be a string, got {type(url).__name__}")
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise InvalidURLError(f"invalid URL {url!r}: {exc}") from exc
    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError(f"only http(s) URLs are allowed, got {parsed.scheme!r}: {url!r}")
    if not parsed.hostname:
        raise InvalidURLError(f"URL has no host: {url!r}")
    if parsed.username is not None or parsed.password is not None:
        raise InvalidURLError(f"URLs with embedded credentials are not allowed: {url!r}")
    return url


# --- Kind inference ----------------------------------------------------------

def infer_kind(name: str, content_type: str | None = None) -> str:
    """Infer a source kind from a content type, then from a filename suffix."""
    if content_type:
        media_type = content_type.lower().split(";")[0].strip()
        if media_type == "text/plain":
            return "txt"
        if media_type in ("text/markdown", "text/x-markdown"):
            return "md"
        if media_type == "application/json":
            return "json"
        if media_type in ("application/x-ndjson", "application/jsonl", "application/x-jsonlines"):
            return "jsonl"
        if media_type in ("text/html", "application/xhtml+xml"):
            return "html"
    suffix = Path(name).suffix.lower()
    if suffix in (".md", ".markdown"):
        return "md"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in (".html", ".htm"):
        return "html"
    return "txt"


# --- Basic text extraction ----------------------------------------------------

def _record(source_id: str, index: int, kind: str, text: str) -> dict:
    record_id = sha256_bytes(f"{source_id}\n{kind}\n{index}".encode("utf-8"))
    return {"record_id": record_id, "kind": kind, "text": text, "source": source_id}


def _extract_text_from_mapping(mapping: Mapping[str, object]) -> str:
    messages = mapping.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            role = message.get("role")
            if isinstance(content, str):
                parts.append(f"{role}: {content}" if isinstance(role, str) else content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    for field in _TEXT_FIELDS:
        value = mapping.get(field)
        if isinstance(value, str) and value:
            return value
    return json.dumps(dict(mapping), ensure_ascii=False, sort_keys=True)


def _record_from_value(value: object, source_id: str, index: int, kind: str) -> dict:
    if isinstance(value, Mapping):
        return _record(source_id, index, kind, _extract_text_from_mapping(value))
    if isinstance(value, str):
        return _record(source_id, index, kind, value)
    return _record(source_id, index, kind, json.dumps(value, ensure_ascii=False, sort_keys=True))


def _extract_jsonl_records(data: bytes, source_id: str) -> list[dict]:
    records: list[dict] = []
    for index, line in enumerate(_decode_text(data).splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestionError(f"invalid JSONL on line {index + 1}: {exc}") from exc
        records.append(_record_from_value(value, source_id, index, "jsonl"))
    return records


def _extract_json_records(data: bytes, source_id: str) -> list[dict]:
    try:
        value = json.loads(_decode_text(data))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"invalid JSON: {exc}") from exc
    if isinstance(value, list):
        return [_record_from_value(item, source_id, index, "json") for index, item in enumerate(value)]
    return [_record_from_value(value, source_id, 0, "json")]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _IGNORED_TAGS:
            self._ignore_depth += 1
        elif self._ignore_depth == 0 and tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)
        elif self._ignore_depth == 0 and tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            self._parts.append(data)


def _extract_html_text(data: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(_decode_text(data))
    parser.close()
    lines = (line.strip() for line in "".join(parser._parts).splitlines())
    return "\n".join(line for line in lines if line)


def extract_text(data: bytes, kind: str, *, source_id: str) -> list[dict]:
    """Extract basic text records from *data* according to *kind*.

    ``kind`` is one of ``txt``, ``md``, ``jsonl``, ``json``, or ``html``. Every
    returned record carries a stable content-addressed ``record_id`` and a
    ``text`` field. The original bytes are never modified; decoding is the only
    lossless step and invalid UTF-8 fails closed.
    """
    kind = kind.lower()
    if kind == "txt":
        return [_record(source_id, 0, "text", _decode_text(data))]
    if kind == "md":
        return [_record(source_id, 0, "markdown", _decode_text(data))]
    if kind == "jsonl":
        return _extract_jsonl_records(data, source_id)
    if kind == "json":
        return _extract_json_records(data, source_id)
    if kind == "html":
        return [_record(source_id, 0, "html", _extract_html_text(data))]
    raise UnsupportedKindError(f"unsupported source kind {kind!r}")


# --- Local file and HTTP retrieval --------------------------------------------

def read_local_file(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[bytes, dict]:
    """Safely read a local ``.txt``/``.md``/``.jsonl`` file with a byte limit.

    Returns ``(raw_bytes, metadata)``. Symlinks are resolved and the extension
    and size are validated before reading.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() not in ALLOWED_LOCAL_EXTENSIONS:
        allowed = ", ".join(ALLOWED_LOCAL_EXTENSIONS)
        raise UnsupportedKindError(f"unsupported extension {resolved.suffix!r}; allowed: {allowed}")
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ContentTooLargeError(f"{resolved.name} is {size} bytes, exceeding the {max_bytes} byte limit")
    data = resolved.read_bytes()
    if len(data) > max_bytes:
        raise ContentTooLargeError(f"{resolved.name} exceeds the {max_bytes} byte limit")
    return data, {"name": resolved.name, "path": str(resolved)}


def _read_limited(stream, max_bytes: int, deadline: float, timeout: float) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() >= deadline:
            raise FetchTimeoutError(f"fetch exceeded the {timeout:g} second time limit")
        want = min(_READ_CHUNK, max_bytes - total + 1)
        chunk = stream.read(want)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise ContentTooLargeError(f"content exceeds the {max_bytes} byte limit")
        chunks.append(chunk)


def fetch_http(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    approved: bool = False,
    user_agent: str = _USER_AGENT,
) -> tuple[bytes, dict]:
    """Fetch *url* over HTTP(S) with size and time limits, only when approved.

    Returns ``(raw_bytes, metadata)`` where metadata includes ``final_url``
    (after redirects), ``status``, and ``content_type``. Both a total time
    budget and a byte budget are enforced; exceeding either fails closed.
    """
    if not approved:
        raise FetchNotApprovedError(f"fetch of {url!r} requires explicit user approval")
    validate_http_url(url)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/plain, text/markdown, text/html, application/json, application/x-ndjson;q=0.9, */*;q=0.1",
        },
    )
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status if hasattr(response, "status") else 200
            final_url = response.geturl()
            content_type = response.headers.get_content_type() if hasattr(response.headers, "get_content_type") else None
            data = _read_limited(response, max_bytes, deadline, timeout)
    except (FetchTimeoutError, ContentTooLargeError):
        raise
    except (TimeoutError, socket.timeout):
        raise FetchTimeoutError(f"fetch of {url!r} timed out") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise FetchTimeoutError(f"fetch of {url!r} timed out") from exc
        raise IngestionError(f"fetch failed for {url!r}: {exc}") from exc
    return data, {"final_url": final_url, "status": status, "content_type": content_type}


# --- Imported source record ----------------------------------------------------

@dataclass(frozen=True)
class ImportedSource:
    source: str
    sha256: str
    bytes: int
    kind: str
    content_type: str | None
    license: str
    retrieval: str
    fetched_at: str | None
    status: int | None
    records: tuple[dict, ...]

    def to_provenance(self) -> dict:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "kind": self.kind,
            "content_type": self.content_type,
            "license": self.license,
            "retrieval": self.retrieval,
            "fetched_at": self.fetched_at,
            "status": self.status,
            "record_count": len(self.records),
        }


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    description: str


# --- Workspace service ---------------------------------------------------------

class WorkspaceService:
    """Headless workspace that stages imports and emits deterministic bundles."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.ingest_dir = self.root / "ingest"
        self.bundles_dir = self.root / "bundles"
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

    def import_local(self, path: str | Path, *, license: str, max_bytes: int = DEFAULT_MAX_BYTES) -> ImportedSource:
        data, meta = read_local_file(path, max_bytes=max_bytes)
        return self.import_bytes(
            data,
            source=meta["path"],
            kind=infer_kind(meta["name"]),
            content_type=None,
            license=license,
            retrieval="local",
        )

    def import_url(
        self,
        url: str,
        *,
        license: str,
        approved: bool = False,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> ImportedSource:
        data, meta = fetch_http(url, max_bytes=max_bytes, timeout=timeout, approved=approved)
        return self.import_bytes(
            data,
            source=meta["final_url"],
            kind=infer_kind(meta["final_url"], meta["content_type"]),
            content_type=meta["content_type"],
            license=license,
            retrieval="http",
            fetched_at=_now_iso(),
            status=meta["status"],
        )

    def import_bytes(
        self,
        data: bytes,
        *,
        source: str,
        kind: str,
        content_type: str | None,
        license: str,
        retrieval: str,
        fetched_at: str | None = None,
        status: int | None = None,
    ) -> ImportedSource:
        """Stage raw *data* content-addressed, then extract and record provenance."""
        if not isinstance(license, str) or not license.strip():
            raise LicenseRequiredError(f"license confirmation required before ingesting {source!r}")
        license = license.strip()
        sha = sha256_bytes(data)
        record_dir = self.ingest_dir / sha[:2] / sha
        record_dir.mkdir(parents=True, exist_ok=True)
        raw_path = record_dir / "raw"
        if not raw_path.exists():
            raw_path.write_bytes(data)
        records = tuple(extract_text(data, kind, source_id=sha))
        imported = ImportedSource(
            source=source,
            sha256=sha,
            bytes=len(data),
            kind=kind,
            content_type=content_type,
            license=license,
            retrieval=retrieval,
            fetched_at=fetched_at,
            status=status,
            records=records,
        )
        (record_dir / "provenance.json").write_bytes(_json_bytes(imported.to_provenance()))
        (record_dir / "records.jsonl").write_bytes(b"".join(_json_bytes(record) for record in records))
        with (self.root / "sources.jsonl").open("ab") as stream:
            stream.write(_json_bytes(imported.to_provenance()))
        return imported

    def create_dataset(
        self,
        dataset_id: str,
        imports: ImportedSource | Iterable[ImportedSource],
        *,
        split: str = "train",
        tokenizer_id: str = "unassigned",
        tokenizer_revision: str = "unassigned",
        seed: int = 0,
    ) -> DatasetBundle:
        """Build a deterministic ``.dts`` bundle from staged imports."""
        if isinstance(imports, ImportedSource):
            import_list = [imports]
        else:
            import_list = list(imports)
        if not import_list:
            raise IngestionError("at least one imported source is required")
        if split not in ("train", "validation", "test"):
            raise IngestionError(f"split must be one of train/validation/test, got {split!r}")
        records: list[dict] = []
        sources: set[str] = set()
        licenses: set[str] = set()
        for imported in import_list:
            sources.add(imported.source)
            licenses.add(imported.license)
            records.extend(dict(record) for record in imported.records)
        splits = {"train": [], "validation": [], "test": []}
        splits[split] = records
        output = self.bundles_dir / f"{dataset_id}.dts"
        return create_bundle(
            output,
            dataset_id,
            splits,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            preprocessing_pipeline="spaceslug-ingest",
            preprocessing_revision="phase-1",
            seed=seed,
            sources=sorted(sources),
            licenses=sorted(licenses),
        )

    def list_imports(self) -> list[ImportedSource]:
        """Rehydrate previously staged imports from the append-only ledger."""
        ledger = self.root / "sources.jsonl"
        if not ledger.exists():
            return []
        result: list[ImportedSource] = []
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            provenance = json.loads(line)
            record_dir = self.ingest_dir / provenance["sha256"][:2] / provenance["sha256"]
            records = tuple(
                json.loads(record_line)
                for record_line in (record_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if record_line.strip()
            )
            result.append(
                ImportedSource(
                    source=provenance["source"],
                    sha256=provenance["sha256"],
                    bytes=provenance["bytes"],
                    kind=provenance["kind"],
                    content_type=provenance.get("content_type"),
                    license=provenance["license"],
                    retrieval=provenance["retrieval"],
                    fetched_at=provenance.get("fetched_at"),
                    status=provenance.get("status"),
                    records=records,
                )
            )
        return result


# --- SearXNG-compatible query ---------------------------------------------------

def _string_field(mapping: Mapping[str, object], key: str, default: str = "") -> str:
    value = mapping.get(key)
    return value if isinstance(value, str) else default


def searxng_search(
    query: str,
    *,
    base_url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[SearchResult]:
    """Query a SearXNG-compatible JSON endpoint and return validated results.

    The query endpoint itself is fetched with approval implied by this explicit
    call; search result documents are never fetched automatically. Result URLs
    that fail validation are skipped; a selected URL must be passed to
    :meth:`WorkspaceService.import_url` with ``approved=True``.
    """
    validate_http_url(base_url)
    if not isinstance(query, str) or not query.strip():
        raise IngestionError("query must be a non-empty string")
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    endpoint = f"{base_url.rstrip('/')}/search?{params}"
    data, _meta = fetch_http(endpoint, max_bytes=max_bytes, timeout=timeout, approved=True)
    try:
        payload = json.loads(_decode_text(data))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"SearXNG returned invalid JSON: {exc}") from exc
    results: list[SearchResult] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        raw_url = item.get("url")
        if not isinstance(raw_url, str):
            continue
        try:
            url = validate_http_url(raw_url)
        except InvalidURLError:
            continue
        results.append(
            SearchResult(
                url=url,
                title=_string_field(item, "title"),
                description=_string_field(item, "content") or _string_field(item, "snippet"),
            )
        )
    return results
