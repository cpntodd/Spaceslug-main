# PersistentTiny bounded scalar metrics

`PersistentTinyTrainer.fixed_scalar_metrics(tokens, targets, mask)` is the convenient Python entry point for the native GPU scalar loss/count binding. It deliberately accepts **exactly 128** token, target, and mask values—the retained Tiny capacity is not expanded by this reporting API.

The result contains `loss`, `count`, operation status, and explicit `fixed_scalar_metrics_retention` metadata. If the native ABI is unavailable, the method returns `status: not-run` rather than falling back silently. Capability metadata advertises the ABI and all three retained dimensions as 128 when available.

This is bounded evaluation/reporting only; it does not imply production training or dataset ownership on the device.
