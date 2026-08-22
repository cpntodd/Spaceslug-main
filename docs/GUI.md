# GUI and local service plan

Spaceslug-main is GUI-first but the GUI is a client of a local service. Training, inference, evaluation, and agent jobs must not run inside the UI process.

```text
Desktop shell / web UI
        ↓ local HTTP, IPC, or Unix socket
Spaceslug service
  ├── model registry
  ├── training manager
  ├── inference server
  ├── evaluation runner
  ├── experiment manager
  ├── agent runner
  └── Spaceslug backend
```

## Initial screens

- **Dashboard:** active model, backend/device, memory, jobs, latest metrics, and recent experiments.
- **Models:** import, create, duplicate, quantize, export, activate adapter, compare revisions, and view provenance.
- **Chat:** streaming conversation, model/adapter selection, sampling, context reset, timings, fallback counters, and transcript export.
- **Training:** dataset, mode, steps/time budget, memory limit, start/pause/cancel, loss curves, validation, checkpoints, and resume.
- **Evaluation:** fixed suites, perplexity/loss, chat, coding, tool use, CPU/Vulkan parity, and quantization comparisons.
- **Experiments:** parent revision, one change, budget, isolated run, metrics, keep/reject, and report.
- **Coding agent:** repository/worktree, task, permission mode, proposed diff, tests, logs, and human approval.

## MVP boundary

The first GUI does not need to be a full IDE. It must allow a user to train, chat, inspect metrics, compare versions, run bounded experiments, and review proposed code changes without a command-line workflow.

## Service requirements

- Local-only by default.
- Explicit job IDs and lifecycle states.
- Cancellation and resource limits.
- Structured logs and progress events.
- SQLite metadata plus immutable artifact directories.
- WebSocket or server-sent events for live jobs.
- No hidden cloud handoff.
- Permission prompts before applying source changes.
