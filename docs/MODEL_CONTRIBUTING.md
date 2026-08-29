# Model contribution guide

**Status: Proposed, non-binding extension contract.** The interfaces below describe the information a new model should expose; they are not a promise that a plugin API already exists.

Spaceslug aims to make new models inspectable, reproducible, and useful on constrained hardware. A model contribution may be a native model family, tokenizer, checkpoint adapter, quantization path, LoRA adapter, or a small test fixture.

## Model card essentials

Every model or adapter should include:

- Name, version, origin, license, and redistribution permissions.
- Architecture, tensor shapes, vocabulary/tokenizer, context length, and precision.
- Required backend capabilities and explicit CPU fallback behavior.
- Parameter/trainable-parameter counts and expected memory footprint.
- Dataset provenance, training/adaptation method, and known limitations.
- Deterministic example inputs and expected outputs or tolerances.

## Proposed extension boundary

A future model/plugin registration should be able to provide:

```text
metadata()       identity, provenance, license, capabilities
 tokenizer()      encode/decode and vocabulary contract
 shapes()         tensor, context, and batch requirements
 forward()        reference and backend execution
 trainable_parts() permitted weights/adapters and optimizer state
 checkpoint()     versioned save/restore with checksums
 evaluate()       deterministic fixtures and metrics
```

Until this becomes an implemented API, use existing host/runtime contracts and document any glue code in the contribution itself. Do not hide model-specific assumptions in generic backend code.

## Acceptance path

1. Start with a small CPU/reference fixture.
2. Add deterministic tokenizer, shape, forward, and checkpoint tests.
3. Add backend support only behind explicit capability checks.
4. Compare backend output and loss against the reference.
5. Publish provenance, artifacts, commands, environment, and limitations.
6. Update the [documentation index](INDEX.md) and mark the exact tested scope.

A new model is not “supported” merely because it loads. Loading, inference, adaptation, checkpoint restore, and backend parity are separate claims.

## Checklist

- [ ] Provenance and license are clear.
- [ ] Tokenizer and vocabulary behavior are tested.
- [ ] Shapes, masks, precision, and ownership are documented.
- [ ] CPU/reference tests pass.
- [ ] Checkpoint compatibility and versioning are specified.
- [ ] Backend requirements and fallback behavior are visible.
- [ ] Evaluation includes limitations and failure cases.
