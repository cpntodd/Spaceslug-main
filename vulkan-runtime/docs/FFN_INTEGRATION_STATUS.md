# FFN integration status

The bounded FFN implementation currently has:

- masked and unmasked forward shader primitives;
- input-backward shader primitive with mask handling;
- deterministic unmasked and masked parameter-gradient shader primitives;
- unmasked and mixed-mask GPU forward parity (`rows=3`, `hidden=4`, `intermediate=16`);
- unmasked GPU input-backward parity;
- mixed-mask GPU parameter-gradient parity;
- deterministic CPU forward/backward and masked parameter-gradient reference coverage;
- compile-time `glslc -O` and SPIR-V validation.

It is not yet a graph-owned training stage. Before integration, add finite-difference checks, masked backward executable coverage, nonzero-moment two-step FFN AdamW parity, checkpoint/C ABI tests, and explicit activation/recomputation policy. Then allocate FFN weights/gradients/moments in the graph and combine the FFN stage with RMSNorm in a bounded end-to-end loss-decrease test.

The schema-v4 and complete full-base APIs remain gated. Dataset-resident training, retained backward/optimizer execution, and arbitrary-shape full-base training remain separate unsupported milestones.
