# Arbitrary-shape full-base ABI design

The fixed Tiny graph remains a specialized profile (`H=64`, `V=259`, `Vp=320`, `Tcap=128`, LoRA ranks 4/8). It must not be widened by silently substituting standalone LM-head training. The general API will use a separate opaque model handle and a validated descriptor.

## Descriptor validation

Creation validates 64-bit checked products for every parameter, activation, gradient, and moment allocation; `hidden`, `vocab`, `padded_vocab`, and sequence capacity are nonzero; `padded_vocab >= vocab`; strides are at least logical widths and are alignment-safe; rank is 0, 4, or 8; and all byte sizes fit device-address and allocator limits. Invalid descriptors fail before Vulkan allocation or mutation.

## Kernel routing

Small aligned shapes use existing tile kernels. Tails route to bounds-checked arbitrary-M/N kernels, with explicit row/column strides. No kernel assumes K-even, 64-wide M/N, or a 128-token capacity. Large matrix and reduction dispatches are chunked into watchdog-safe submissions. Scratch buffers are sized from the checked descriptor and remain device-resident between stages.

## Execution and state

A complete step is one ordinary submission chain initially. The control block selects parameter groups and owns rows, masks, optimizer controls, and step. Validation occurs before recording; selected groups either all update or none update. AdamW moments and descriptor metadata are checkpointed together. Retained execution is a later fixed-shape schedule over the same control block, not a second training implementation.

## Test matrix

CPU-double references and Vulkan parity cover M/N in `{1,7,63,64,65,127}`, K in `{1,31,32,33,63,64}`, odd vocabularies, short and maximum sequences, ranks 0/4/8, tail strides, overflow and memory-pressure rejection, watchdog chunking, checkpoint resume, and pure-C callers. Capability exposure remains disabled until both lavapipe and RADV/POLARIS10 pass.
