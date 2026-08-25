# Spaceslug-main — Phase 1 desktop user guide

> **Status: documentation for the Phase 1 desktop surface (modules committed).**
>
> This is the user guide for the Phase 1 desktop surface reached through the
> `spaceslug desktop` command. It describes what is committed and observable in
> the repository as of this writing, and it marks clearly what is still a
> placeholder.
>
> The Tkinter desktop shell (`python/spaceslug/desktop/`), the loopback-only
> OpenAI-compatible service (`python/spaceslug/openai_api.py`), and the headless
> ingestion/workspace service with its approval and SearXNG policy
> (`python/spaceslug/workspace.py`) are committed. Training, chat, and the
> desktop's "Start API" action are still placeholders in the shell, and PDF text
> extraction is not implemented yet. Do not treat any line as a shipped
> capability until the Phase 1 gate in [`STAGED_ROADMAP.md`](STAGED_ROADMAP.md)
> is met.

## What this guide covers

1. [Launching the desktop app](#1-launching-the-desktop-app)
2. [The Tkinter non-browser app](#2-the-tkinter-non-browser-app)
3. [Local files: `.txt`, `.md`, `.jsonl`, `.pdf`](#3-local-files-txt-md-jsonl-pdf)
4. [Approved URLs and SearXNG policy](#4-approved-urls-and-searxng-policy)
5. [GPU-first with explicit fallback](#5-gpu-first-with-explicit-fallback)
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
for the Phase 1 Tiny-model loop (select data, train or adapt, chat, inspect
results) without the command line. The terminal TUI (`spaceslug tui`) remains
available and is not removed by the desktop app.

The desktop app is a **client** of a local Spaceslug service; training,
inference, and evaluation do not run inside the window process. The current
shell displays job/placement state and, where an action is not wired yet, says
so explicitly rather than hiding it (see [`GUI.md`](GUI.md)).

## 2. The Tkinter non-browser app

The Phase 1 desktop app is a **native window built with Tkinter/ttk**, not a web
application:

- **No browser required.** You do not open a URL or a localhost web page to use
  it. It opens an ordinary desktop window titled *"Spaceslug-main — desktop
  shell (Phase 1)"*.
- **Local-only by default.** There is no hidden cloud handoff; models, data, and
  jobs stay on this machine (see the service requirements in [`GUI.md`](GUI.md)).
- **Five tabs.** The window is a notebook with these tabs, mirroring the
  `GUI.md` plan: **Home**, **Datasets**, **Build & Train**, **Interact**, and
  **Local API**.

What each tab shows today:

- **Home** — the fixed Spaceslug-Tiny profile (hidden size, layers, attention
  heads, context length, training mode, dtype, LoRA rank, parameter estimate)
  and a **GPU-primary / CPU fallback status** box (mode, device, reason).
- **Datasets** — a *File path*, *URL*, and *Search* field. These are recorded but
  **not wired** to a dataset service in the current shell.
- **Build & Train** — a steps control and the live **loss worm** canvas, with
  demo controls to record/clear a loss value. Training is not wired; "Start
  training" is a placeholder.
- **Interact** — a prompt field. Streaming chat requires the local service, which
  is not implemented in the shell; "Send" is disabled.
- **Local API** — an address field (default `127.0.0.1:8123`). No server is
  wired into the shell; "Start API" is disabled.

Each placeholder tab displays its **capability boundary as read-only text**, so
the boundary is surfaced to the user instead of hidden.

## 3. Local files: `.txt`, `.md`, `.jsonl`, `.pdf`

The desktop and its headless workspace service import **local documents** for
dataset building and chat context. Phase 1 targets these formats:

| Format | Current status |
|---|---|
| `.txt` | Selected by the file picker **and** readable as text by the workspace service. |
| `.md`  | Selected **and** readable (recorded as markdown). |
| `.jsonl` | Selected **and** readable; each JSONL line becomes one record. |
| `.pdf`  | Selected by the file picker, but PDF→text extraction is **not implemented yet**. |

Details:

- **Text-first.** `.txt`, `.md`, and `.jsonl` are already text and can be
  tokenized directly. `.pdf` is a binary document; selecting it is supported, but
  extracting its text is still pending — no PDF extractor exists in the
  ingestion service yet.
- **Ingestion stays local and inspectable.** The workspace service reads local
  files with a per-file size limit, stages them content-addressed (SHA-256),
  requires a license confirmation before ingest, and builds a deterministic,
  versioned dataset bundle (`.dts`) with lineage and checksums. See
  [`DATASETS.md`](DATASETS.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
- **Reproducibility over filenames.** A dataset is identified by content hash and
  lineage, not by a mutable filename. Raw text is not silently replaced by
  summaries or truncation; invalid UTF-8 fails closed.

## 4. Approved URLs and SearXNG policy

The workspace service implements a **fetch-only-with-approval** policy for web
content, with SearXNG as the search gateway:

- **Explicit approval before any fetch.** An HTTP(S) fetch refuses to run unless
  it was explicitly approved. An unapproved URL is a hard error, not a silent
  fetch or a silent drop.
- **URL validation.** Only `http`/`https` URLs with a host are accepted; URLs
  with embedded credentials are rejected.
- **Search via SearXNG.** Web search queries a SearXNG-compatible JSON endpoint
  (`format=json`). SearXNG returns result URLs/titles/descriptions only — it
  **never auto-fetches a result document**. A result URL must be passed back
  through the import path with explicit approval to be retrieved.
- **License confirmation.** A source cannot be ingested without a license
  confirmation, local or remote alike.
- **Limits.** Fetches enforce a byte budget and a time budget; exceeding either
  fails closed.
- **Desktop status.** The *URL* and *Search* fields on the **Datasets** tab are
  placeholders: they are recorded but not wired to this service in the current
  shell. The policy lives in the headless workspace service, not yet in the
  window.

## 5. GPU-first with explicit fallback

The desktop is expected to prefer the GPU within a bounded, supported path, and
to make any fallback visible rather than silent.

- **GPU is primary only for explicitly supported native operations.** The GPU
  (Vulkan/RX580) is used when an operation is inside the supported, validated
  contract — for the fixed Tiny profiles, the documented graph-owned operations.
  See [`STAGED_ROADMAP.md`](STAGED_ROADMAP.md) and
  [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md).
- **CPU is the fallback outside that path.** Work outside the GPU contract runs
  on the CPU reference backend, which remains authoritative for correctness.
- **Fallback is explicit, never silent** ([`DECISIONS.md`](DECISIONS.md), D005).
  The Home tab shows a placement status computed from a real capability probe:
  `gpu-primary` (a real device, not software Vulkan, and a GPU operation
  present) or `cpu-fallback`, plus a reason. `gpu-primary` is reported only when
  a hardware GPU path actually exists; software Vulkan (lavapipe) is reported as
  CPU fallback, not as GPU primary.
- **CPU is always available.** Even in `gpu-primary` mode the CPU reference
  remains the recorded fallback, and any fallback that occurs is captured in
  metrics rather than hidden.

## 6. The live loss worm

During training, the desktop shows a **live loss worm**: a compact, rolling
curve that plots one loss point per step, so you can watch convergence or
divergence without waiting for the run to finish.

- **Live and rolling.** Each recorded step appends a loss value; the worm keeps a
  bounded window (64 points by default) and scrolls older points off the left.
- **Rendered on a canvas.** The worm is a connected line with a marker per point
  on a dark `tkinter.Canvas`, with lower loss drawn higher.
- **Current shell behavior.** The worm is controller-driven and render-safe: in
  the shell it is fed by a demo *Record loss* control, and it is exercised
  headlessly by tests. When real training is wired, the same graph is fed one
  point per step.
- **The worm is a view, not the record.** Authoritative loss values land in run
  metrics and the experiment record, tied to the exact dataset, model,
  tokenizer, code, and configuration revisions (see
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

In practice: train → checkpoints are written while training and a checksummed
artifact plus an experiment record are emitted at the end. Keep the checkpoint
to resume; keep the artifact to load the model for chat; the experiment record
is your evidence for a claim.

## 8. Local OpenAI-compatible API

A **local, loopback-only OpenAI-compatible HTTP service** is implemented so
tools that already speak the OpenAI protocol (editors, assistants, scripts) can
point at this machine instead of a cloud endpoint.

- **Endpoints.** `GET /health`, `GET /v1/models`, and
  `POST /v1/chat/completions`, returning OpenAI-shaped JSON (`chat.completion`,
  `choices[].message`, `usage`, `system_fingerprint`).
- **Loopback only.** The server binds to `127.0.0.1`/`::1`/`localhost` and
  refuses non-loopback hosts — it is not reachable from other machines.
- **Non-streaming only.** Streaming completions (`"stream": true`) are rejected
  explicitly rather than half-supported.
- **Truthful, non-generative default.** The default responder runs the current
  CPU Tiny attention reference and returns an echo-safe message that names its
  backend (`cpu-tiny`) and model (`spaceslug-tiny-attention-v1`) and reports a
  single argmax next-token prediction — it does not fabricate natural-language
  completions.
- **Status.** The server exists as a module with tests, but it is **not yet wired
  to a CLI command or to the desktop's "Start API" button**, which is disabled.

## Status and blockers

Committed and observable: the Tkinter desktop shell and its `spaceslug desktop`
entry point (`fd04de4` "Add native Spaceslug desktop shell"), the loopback
OpenAI-compatible server (`5aaa52a` "Add local OpenAI-compatible Tiny API"), the
headless workspace/ingestion service with its approval + SearXNG policy
(`0dbc7d7` "Add approved dataset workspace ingestion"), the canvas loss worm,
and the GPU-primary/CPU-fallback placement status. PDF is in the file-picker
selection (`e026ba1` "Include PDFs in dataset file selection"), alongside
`.txt`/`.md`/`.jsonl`.

Still missing or placeholder as of this writing:

- **PDF text extraction** is not implemented — `.pdf` files are selected by the
  picker but are not readable as text by the ingestion service.
- The desktop **Datasets URL/Search**, **Interact Send**, and **Local API Start**
  controls are placeholders, not wired to the (committed) headless services.
- No CLI command launches the OpenAI-compatible server, and the desktop does not
  start it.

This guide is documentation only; it makes no changes to code or module
contracts.

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
