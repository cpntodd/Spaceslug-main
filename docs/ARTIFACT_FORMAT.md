# Spaceslug artifact format

The artifact format is backend-independent and versioned. Raw Cactus bundles, Colibri directories, and Hugging Face checkpoints are importer inputs, not the long-term runtime contract.

## Proposed layout

```text
spaceslug-model/
├── manifest.json
├── model.json
├── tokenizer/
├── graph/
│   ├── prefill.plan
│   ├── decode.plan
│   └── train.plan
├── tensors/
│   ├── index.json
│   ├── dense/
│   └── experts/
├── quantization/
├── adapters/
└── checksums/
```

## Manifest requirements

The manifest must identify:

- schema version;
- model and architecture name;
- base-model provenance;
- tokenizer and chat-template revisions;
- vocabulary size and special tokens;
- tensor dtype, layout, orientation, and alignment;
- quantization format, group size, scales, and transforms;
- hidden/layer/attention/KV dimensions;
- RoPE parameters;
- tied embedding/lm-head behavior;
- MoE router semantics and expert ordering;
- supported execution modes;
- adapter compatibility;
- checksums and source license/provenance.

## Compatibility rules

- A LoRA adapter must declare the exact base-model identity and target tensor paths.
- A graph plan must declare its required operator capabilities.
- A tokenizer mismatch is a load error, not a warning.
- Unknown schema versions fail closed.
- Quantization metadata is never inferred from filenames alone.
- Storage or placement changes must not alter router or tokenizer semantics.

## Training artifacts

Training checkpoints must include:

- model weights or immutable base reference;
- trainable parameters;
- optimizer state;
- scheduler state;
- RNG state;
- dataset and preprocessing revisions;
- training configuration;
- code revision;
- validation metrics;
- parent checkpoint.

## Adapter artifacts

Adapters must include:

- adapter schema version;
- base-model fingerprint;
- target module paths;
- rank and alpha;
- dtype;
- parameter tensors;
- tokenizer/chat-template compatibility;
- training provenance;
- optional merge/export metadata.
