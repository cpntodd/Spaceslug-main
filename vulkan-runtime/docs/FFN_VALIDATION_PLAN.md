# FFN validation plan

The bounded FFN now has deterministic CPU forward/backward reference coverage and a registered finite-difference check for the W1 analytic gradient:

- `tests/test_ffn_cpu_reference.cpp`
- `tests/test_ffn_finite_difference.cpp`

The finite-difference gate uses central differences (`h=1e-3`) on the small `rows=3`, `hidden=4`, `intermediate=16` contract and passes with maximum error near `1.03e-6`.

GPU parity exists for forward (unmasked and mixed-mask), input backward, and mixed-mask parameter gradients. Remaining integration gates are finite differences for additional parameter classes, masked backward executable coverage, nonzero-moment two-step FFN AdamW parity, checkpoint/C ABI tests, and graph-owned FFN state.
