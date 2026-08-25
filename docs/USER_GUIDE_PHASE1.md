# Spaceslug-main — Phase 1 desktop user guide

> **Status: documentation for the committed Phase 1 desktop application.**
>
> This is the user guide for the Phase 1 desktop surface reached through the
> `spaceslug desktop` command. It describes the committed, working desktop
> application and marks clearly what is still a bounded limitation rather than a
> hidden one.
>
> The Tkinter desktop shell (`python/spaceslug/desktop/`), the loopback-only
> OpenAI-compatible service (`python/spaceslug/openai_api.py`), and the headless
> ingestion/workspace service with its approval and SearXNG policy
> (`python/spaceslug/workspace.py`) are committed and wired into the desktop.
> Remaining boundaries — PDF text extraction, CPU-only Phase 1 training, the
> non-generative chat responder, and non-streaming API — are surfaced in the app
> and recorded here. See [`STAGED_ROADMAP.md`](STAGED_ROADMAP.md) for the
> authoritative fixed Tiny boundaries.

## What this guide covers

1. [Launching the desktop app](#1-launching-the-desktop-app)
2. [The Tkinter non-browser app](#2-the-tkinter-non-browser-app)
3. [Local files: `.txt`, `.md`, `.jsonl`](#3-local-files-txt-md-jsonl)
4. [Approved URLs and SearXNG policy](#4-approved-urls-and-searxng-policy)
5. [GPU-primary with explicit fallback](#5-gpu-primary-with-explicit-fallback)
6. [The live loss worm](#6-the-live-loss-worm)
7. [Checkpoint artifacts](#7-checkpoint-artifacts)
8. [Local OpenAI-compatible API](#8-local-openai-compatible-api)

## 1. Launching the desktop app

Run:

```text
spaceslug desktop
```

Optional arguments:

```text
spaceslug desktop --runtime-root /path/to/vulkan-runtime --runtime-revision runtime
```

`spaceslug desktop` starts a native desktop application — the GUI-first surface
for the Phase 1 Tiny-model loop (import data, train, chat, inspect results)
without the command line. The terminal TUI (`spaceslug tui`) remains available
and is not removed by the desktop app.

The desktop app is a **client** of the local Spaceslug modules. The only
long-running work — CPU projected Tiny training — runs in a background worker
thread owned by the controller; the window polls a thread-safe loss queue on the
main thread so the interface stays responsive. Dataset import, URL import, and
SearXNG search are user-initiated, time-bounded calls.

## 2. The Tkinter non-browser app

The Phase 1 desktop app is a **native window built with Tkinter/ttk**, not a web
application:

- **No browser required.** You do not open a URL or a localhost web page to use
  it. It opens an ordinary desktop window titled *"Spaceslug-main — desktop
  (Phase 1)"*.
- **Local-only by default.** There is no hidden cloud handoff; models, data, and
  jobs stay on this machine (see the service requirements in [`GUI.md`](GUI.md)).
- **Five tabs.** The window is a notebook with these tabs, mirroring the
  `GUI.md` plan: **Home**, **Datasets**, **Build & Train**, **Interact**, and
  **Local API**.

What each tab does:

- **Home** — the fixed Spaceslug-Tiny profile (hidden size, layers, attention
  heads, context length, training mode, dtype, LoRA rank, parameter estimate)
  and a **GPU-primary / CPU fallback status** box showing *requested*, *actual*,
  *hardware*, and a reason.
- **Datasets** — the wired ingestion workflow: a required license field, local
  source import (`.txt`/`.md`/`.jsonl`), approved-URL import, SearXNG search and
  result selection, and `.dts` bundle creation.
- **Build & Train** — training controls (steps, learning rate, batch size),
  start/stop, and the live **loss worm** canvas with a status readout (run id,
  checkpoint/artifact/experiment paths, loss trajectory, placement).
- **Interact** — a chat prompt with a transcript, backed by the committed CPU
  Tiny echo responder.
- **Local API** — an address field (default `127.0.0.1:8123`) with Start/Stop
  controls for the loopback OpenAI-compatible server.

Each tab also shows its **capability boundary as read-only text**, assembled
from the live module capability metadata, so the boundary is surfaced to the
user instead of hidden.

## 3. Local files: `.txt`, `.md`, `.jsonl`

The desktop imports **local text documents** for dataset building and chat
context. Phase 1 imports these formats:

| Format | Status |
|---|---|
| `.txt` | Imported as text. |
| `.md`  | Imported as markdown text. |
| `.jsonl` | Imported; each JSONL line becomes one record. |

Details:

- **Text-first.** These are already text and can be tokenized directly.
- **PDF is not importable yet.** The underlying file picker can select `.pdf`
  files, but PDF-to-text extraction is not implemented, so the desktop's local
  source chooser is `.txt`/`.md`/`.jsonl` only. This is a recorded limitation,
  not a hidden one.
- **Ingestion stays local and inspectable.** The workspace service reads local
  files with a per-file size limit, stages them content-addressed (SHA-256),
  requires a license confirmation before ingest, and builds a deterministic,
  versioned dataset bundle (`.dts`) with lineage and checksums. See
  [`DATASETS.md`](DATASETS.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
- **Reproducibility over filenames.** A dataset is identified by content hash and
  lineage, not by a mutable filename. Raw text is not silently replaced by
  summaries or truncation; invalid UTF-8 fails closed.

## 4. Approved URLs and SearXNG policy

The desktop's **Datasets** tab drives a **fetch-only-with-approval** policy for
web content, with SearXNG as the search gateway:

- **License first.** A source cannot be imported — local or remote — without a
  license confirmation entered in the Datasets tab.
- **Explicit approval before any fetch.** The "Import URL" action treats the URL
  as approved; an HTTP(S) fetch refuses to run unless it was explicitly approved.
  An unapproved URL is a hard error, not a silent fetch or a silent drop.
- **URL validation.** Only `http`/`https` URLs with a host are accepted; URLs
  with embedded credentials are rejected.
- **Search via SearXNG.** The "Search" action queries a SearXNG-compatible JSON
  endpoint (configurable base URL, default `http://127.0.0.1:8888`). SearXNG
  returns result URLs/titles/descriptions only — it **never auto-fetches a
  result document**. You select a result, then "Import selected" fetches it
  through the approved-URL path.
- **Limits.** Fetches enforce a byte budget and a time budget; exceeding either
  fails closed and reports an error in the tab's status line.

## 5. GPU-primary with explicit fallback

The desktop prefers the GPU within a bounded, supported path, and makes any
fallback visible rather than silent.

- **GPU is primary only for explicitly supported native operations.** The GPU
  (Vulkan/RX580) is used when an operation is inside the supported, validated
  contract — for the fixed Tiny profiles, the documented graph-owned operations.
  See [`STAGED_ROADMAP.md`](STAGED_ROADMAP.md) and
  [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md).
- **Training placement is explicit: requested vs actual.** The Home and
  Build & Train tabs show both *requested* (what you asked for: `gpu-primary` or
  `cpu`) and *actual* (what the job really runs on). In Phase 1, dataset-integrated
  projected Tiny training always resolves to `cpu-fallback` — the CPU reference
  is authoritative — even when a hardware GPU is present, and the reason is shown.
- **Fallback is explicit, never silent** ([`DECISIONS.md`](DECISIONS.md), D005).
  Software Vulkan (lavapipe) is reported as CPU fallback, not as GPU primary, and
  any fallback that occurs is captured in metrics rather than hidden.
- **CPU is always available.** Even in `gpu-primary` mode the CPU reference
  remains the recorded fallback.

## 6. The live loss worm

During training, the Build & Train tab shows a **live loss worm**: a compact,
rolling curve that plots one loss point per step as it happens.

- **Live and rolling.** Training runs in a background worker thread and pushes
  each step's loss through a thread-safe queue; the main thread polls it (every
  200 ms) and redraws the worm. The worm keeps a bounded window (64 points) and
  scrolls older points off the left.
- **Rendered on a canvas.** The worm is a connected line with a marker per point
  on a dark `tkinter.Canvas`, with lower loss drawn higher.
- **Cancelable.** The Stop button requests cooperative cancellation; the worker
  checks the cancel flag between steps and records `stopped_reason: cancelled`.
- **The worm is a view, not the record.** Authoritative loss values land in the
  training status readout and the experiment record, tied to the exact dataset,
  model, tokenizer, code, and configuration revisions (see
  [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). Read the worm for live intuition;
  rely on recorded metrics for comparisons.

## 7. Checkpoint artifacts

Training produces two distinct kinds of files, plus a run record. They are
different and are not interchangeable:

- **Checkpoint** — resumable training state: model weights (or an immutable base
  reference), trainable parameters, optimizer state, scheduler state, RNG state,
  the training configuration, and the parent checkpoint. A checkpoint lets you
  pause and resume a run exactly. Checkpoints are schema-versioned and fail
  closed on unknown versions or incompatible resumes (dataset revision,
  tokenizer fingerprint, batch size, optimizer, or weight decay must match).
- **Artifact** — a versioned, checksummed model or adapter bundle for *use*
  (inference and reuse), not for resuming training. An artifact carries its
  manifest, tokenizer, tensors, quantization metadata, adapter compatibility,
  checksums, and provenance. See [`ARTIFACT_FORMAT.md`](ARTIFACT_FORMAT.md).
- **Experiment record** — the machine-readable metadata for a run (identity
  fields, metrics, fallbacks, and a reference to the large checkpoint/artifact
  outputs). It is what makes a result reproducible and auditable.

In practice: train → the Build & Train status shows the resolved `checkpoint`,
`artifact`, and `experiment` paths, and the loss trajectory. Keep the checkpoint
to resume; keep the artifact to load the model; the experiment record is your
evidence for a claim.

## 8. Local OpenAI-compatible API

The desktop's **Local API** tab starts and stops a **local, loopback-only
OpenAI-compatible HTTP service**, so tools that already speak the OpenAI
protocol (editors, assistants, scripts) can point at this machine instead of a
cloud endpoint.

- **Endpoints.** `GET /health`, `GET /v1/models`, and
  `POST /v1/chat/completions`, returning OpenAI-shaped JSON (`chat.completion`,
  `choices[].message`, `usage`, `system_fingerprint`).
- **Loopback only.** The server binds to `127.0.0.1`/`::1`/`localhost` and
  refuses non-loopback hosts — it is not reachable from other machines.
- **Non-streaming only.** Streaming completions (`"stream": true`) are rejected
  explicitly rather than half-supported.
- **Truthful, non-generative default.** The served responder runs the current
  CPU Tiny attention reference and returns an echo-safe message that names its
  backend (`cpu-tiny`) and model (`spaceslug-tiny-attention-v1`) and reports a
  single argmax next-token prediction — it does not fabricate natural-language
  completions. The same responder backs the Interact tab.

## Status and boundaries

Committed and wired: the Tkinter desktop shell with its `spaceslug desktop`
entry point, the five-tab workflow (import → `.dts` → train → chat → local API),
the canvas loss worm fed live by the training worker thread, the GPU-primary
requested/actual placement status, the loopback OpenAI-compatible server, and
the headless workspace/ingestion service with its approval + SearXNG policy.

Remaining boundaries (all surfaced in the app, none hidden):

- **PDF text extraction** is not implemented — `.pdf` files are selectable by
  the underlying picker but not importable as text yet.
- **Training is CPU-only in Phase 1.** Dataset-integrated GPU training is not
  available; requesting `gpu-primary` records the intent but the actual
  placement resolves to `cpu-fallback`.
- **Chat is non-generative.** The responder is an echo-safe CPU reference, not a
  natural-language generator.
- **The API is non-streaming** and serves the same non-generative responder.

## Related documents

- [`STAGED_ROADMAP.md`](STAGED_ROADMAP.md) — approved sequence; Phase 1 gate.
- [`GUI.md`](GUI.md) — GUI and local service plan.
- [`PRODUCT.md`](PRODUCT.md) — product definition and MVP user journey.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — module layout and backend contract.
- [`DECISIONS.md`](DECISIONS.md) — D005 explicit fallback, among others.
- [`DATASETS.md`](DATASETS.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md),
  [`ARTIFACT_FORMAT.md`](ARTIFACT_FORMAT.md) — data, measurement, and artifact
  contracts.
- [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md) — fixed Tiny
  boundaries and fallback semantics.
