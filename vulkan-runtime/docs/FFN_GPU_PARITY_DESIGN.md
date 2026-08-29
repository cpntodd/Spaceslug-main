# FFN GPU parity design

The clean unmasked diagnostic shader `ffn_gelu_forward_unmasked.comp` isolates FFN arithmetic from row-mask descriptor behavior. Its contract is six storage buffers (`X`, `W1`, `B1`, `W2`, `B2`, `Y`) and push constants `{rows, hidden, intermediate}`. The existing masked production primitive remains separate and requires the seventh mask binding.

The next parity executable should use small dimensions (`rows=3`, `hidden=4`, `intermediate=16`), independent device-local buffers, explicit ranges, and one dedicated staging readback region. Compare forward output against the CPU reference, then repeat with mixed masks using the masked shader. Only after forward parity should backward and parameter-gradient harnesses be added.
