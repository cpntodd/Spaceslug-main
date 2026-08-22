# Devlog 0006 — Runtime-owned Vulkan vector API

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revision:** pending host documentation commit
- **vulkan-runtime revision:** `a195bc6`

## Objective

Replace the reference-only native tensor function with a runtime-owned shared library entry point that executes the existing Vulkan vector-add path and returns device-produced output.

## Result

`vulkan-runtime` now builds `libvulkan_runtime_api.so` and exports:

```cpp
extern "C" int spaceslug_vector_add(
    float const* in_a,
    float const* in_b,
    float* out,
    std::size_t n
);
```

The implementation reuses the validated embedded `vector_add` shader flow, accepts the established one-million-element contract, copies the mapped device output into the caller's buffer, and returns an error for invalid pointers or shape.

## Evidence

- Runtime configure/build passed.
- Runtime shared library linked successfully after enabling PIC on the core library.
- Direct client passed on RADV:
  `runtime_api vector_add: PASS`.
- Direct client passed with lavapipe:
  `runtime_api vector_add: PASS`.
- Full runtime CTest passed on RADV: 25/25.
- Full runtime CTest passed with lavapipe: 25/25.

## Limitation

The host repository currently contains the ABI declaration and client source, but does not yet automate cross-repository shared-library discovery or loading. That is the next integration task. No benchmark claim is made.
