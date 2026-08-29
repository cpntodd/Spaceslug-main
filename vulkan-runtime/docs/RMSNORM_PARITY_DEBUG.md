# RMSNorm backward parity debug plan

The clean identity-copy diagnostic now demonstrates that the staging, descriptor, submission, and dedicated output readback path can pass independently. The prior RMSNorm mismatch came from a malformed/stale diagnostic executable path; it must not be used as mathematical evidence.

Next, rebuild the backward test from the identity helper with four explicit bindings and `rmsnorm_backward.spv`, then compare CPU-double results. Keep output in its own device allocation and a dedicated staging range. Register only after lavapipe and RADV pass.
