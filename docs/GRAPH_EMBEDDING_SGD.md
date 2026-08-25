# Graph-owned embedding SGD

Spaceslug optionally binds `spaceslug_tiny_forward_train_embeddings_sgd` when the
runtime exports it. The binding is integrated graph training: activations,
dstate gradients, token buffers, and the embedding update are owned by the
persistent Tiny graph for one normal submission. It is deliberately distinct
from the standalone `spaceslug_embedding_training_*` API, which consumes
caller-supplied dstate rows.

The trainer accepts one host-staged window with equal token, target, and mask
arrays and `1 <= rows <= 128`. The dataset remains host-owned; no dataset batch
or retained training buffer is persisted on the device. The result metadata
reports `graph_owned_embedding: true`, `standalone_api: false`, and
`dataset_device_resident: false`.

Only graph-owned token embeddings are covered. Positions, feed-forward weights,
and normalization remain unsupported and are not silently included. AdamW is
not part of this ABI; the operation is sparse FP32 SGD. If the runtime symbol
is absent, capability metadata reports status `unsupported` and return code
`-5`, without falling back to the standalone embedding API.

See the runtime capability string and `tests/test_native_training.py` for the
binding and boundary checks.

> **Note:** The native runtime already enforces token/target/mask validity and
> the fixed `rows <= 128` window. The Python binding repeats the shape and
> learning-rate validation before crossing the ABI.

> **Status:** Experimental; CPU reference remains authoritative outside the
> optional graph-owned runtime path.

## Boundary summary

| Area | Graph-owned embedding SGD |
|---|---|
| Optimizer | FP32 SGD |
| Rows | 1–128 |
| Dataset ownership | Host |
| Retained training buffers | Unsupported / host boundary preserved |
| Positions | Unsupported |
| FFN | Unsupported |
| Normalization | Unsupported |
| Standalone embedding API | Separate; never used as fallback |

The runtime ABI is optional so older libraries continue to load and report an
explicit unsupported capability rather than claiming integrated training.

## Verification

```bash
PYTHONPATH=python python3 -m unittest -v
```

The full Spaceslug-main unittest suite includes mocked ABI tests for symbol
binding, row validation, and metadata boundaries.

## Commit

This milestone is committed in the Spaceslug-main repository after the full
unittest suite passes.

## Scope note

The ABI is graph-owned and therefore does not alter the standalone embedding
training contract or its caller-supplied dstate layout.

## Safety note

No host readback bridge is introduced: the graph computes dstate gradients and
applies sparse SGD before completion of the normal submission.

## Compatibility

Capability discovery is independent of the standalone API and exposes both
contracts so callers can choose explicitly.

## Future work

Embedding checkpoint import/readback and additional parameter groups require a
separate ABI milestone; positions, FFN, and normalization must remain explicit
until their graph ownership and synchronization contracts exist.

## Test intent

The regression tests ensure that a runtime mock receives the exact row count
and learning rate, while oversized windows fail before native invocation.

## Dataset boundary

Host callers may stream dataset windows through the trainer, but the runtime
operation itself receives only the current bounded arrays and retains no
whole-dataset state.

## Retained boundary

Retained forward/loss APIs remain execution-only. They do not become training
or optimizer storage as a consequence of this ABI.

## Metadata

The capability record identifies `parameter_group: embeddings`,
`integrated_graph_sgd: true` when available, `standalone_api: false`, and the
unsupported group flags for positions, FFN, and normalization.

## End state

This document intentionally describes an optional integration, not full-base
training.

## Ownership

The graph owns only the embedding update state required by this operation;
the host owns data selection, window boundaries, and checkpoint orchestration.

## Failure mode

Missing native symbols produce a clear backend error and capability status
rather than a silent CPU or standalone fallback.

## ABI shape

The native call is `(graph, tokens, targets, masks, rows, learning_rate)` and
returns zero on success.

## Review checklist

- rows validated before ctypes call
- graph and standalone metadata distinct
- positions/FFN/norm unsupported
- dataset and retained boundaries preserved
- full unittest run before commit

## End

The compact summary above is the normative contract for this milestone.

## Appendix

No additional tensors are accepted by the Python binding; graph-owned dstate
is intentionally not exposed as a host input.

## Versioning

Future incompatible changes require a new capability string and schema update.

## Support

Use the runtime capability query to detect availability before selecting this
trainer method.

## Non-goal

This does not claim arbitrary model shapes or unrestricted embedding training.

## Final boundary

The standalone sparse embedding API remains available independently and is not
wrapped or reused by this integrated operation.

## EOF

End of document.

