# Supported Ops

This document lists currently supported ops and their backend coverage.

## Backend overview

- **CPU:** scalar Rust kernels.
- **CPU (AVX/AVX2):** SIMD kernels for x86_64 when enabled.
- **Vulkan:** compute kernels compiled from Slang with shader-side casting.

## Adding custom ops

This is the minimal checklist to add a new op end-to-end.

### CPU

1. Add the op to `openinfer/src/graph.rs`:
   - `OpKind` variant and `as_str()` mapping.
   - Add any new attributes to `OpAttrs` if needed.
2. Add a CPU implementation under `openinfer/src/ops/cpu/<op>/`:
   - `mod.rs`, `<op>.rs`, `registry.rs`.
   - Optional inplace kernels: `<op>_inplace.rs` + `registry_inplace.rs`.
3. Register the op in the CPU registry:
   - `openinfer/src/ops/cpu/registry.rs` for out-of-place.
   - `openinfer/src/ops/cpu/registry_inplace.rs` via the op module.
4. If the op supports broadcasting, update the policy in
   `openinfer/src/ops/registry.rs` (`broadcast_policy`).

### CPU AVX / AVX2

1. Add SIMD implementations under:
   - `openinfer/src/ops/cpu_avx/<op>/`
   - `openinfer/src/ops/cpu_avx2/<op>/`
2. Mirror the CPU file layout:
   - `mod.rs`, `<op>.rs`, `registry.rs`, optional `<op>_inplace.rs`,
     `registry_inplace.rs`, `registry_accumulate.rs`.
3. Register the kernels in the per-device registries:
   - `openinfer/src/ops/cpu_avx/registry.rs`
   - `openinfer/src/ops/cpu_avx2/registry.rs`
4. If a dtype is not possible with SIMD, route it to CPU fallback in the
   registry (AVX/AVX2 should still advertise CPU-equivalent coverage).

### Vulkan

1. Add the op under `openinfer/src/ops/vulkan/<op>/`:
   - `mod.rs`, `registry.rs`, `registry_inplace.rs` (if supported).
   - Per-dtype shaders under `shaders/{base,inplace,accumulate}/`.
2. Add an entry to `openinfer/src/ops/vulkan/shaders.json` with `shader_dir`
   and `spv_dir` so `build.rs` compiles and embeds SPIR-V.
   (Broadcast is a backend-only exception.)
3. Implement target selection in `openinfer/src/ops/vulkan/<op>/mod.rs`
   and add it to `openinfer/src/ops/vulkan/mod.rs` dispatch.
4. Register the Vulkan kernel in `openinfer/src/ops/vulkan/<op>/registry.rs`.
5. If you add an inplace variant, wire it in
   `openinfer/src/ops/vulkan/<op>/registry_inplace.rs`.
6. Ensure dtype constraints are enforced in the Vulkan backend
   (`openinfer/src/backend/vulkan/mod.rs`).

### Inplace / Accumulation variants

- If the op supports in-place execution, add `<op>_inplace.rs` and wire it
  in `registry_inplace.rs` for each backend.
- If the op supports `acc=`, add `<op>_accumulate.rs` and
  `registry_accumulate.rs` for each backend.
- Accumulation kernels accept an optional output buffer to support reuse.

## Op coverage

The tables below list support by device. Each row includes input/output shapes,
attributes, dtype coverage (grouped by float/signed/unsigned/packed), and
device-specific notes.

---

| Op name | Dtypes | Input | Output | Attributes | Inplace | Accumulation | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise add with broadcasting. Accumulate is integer-only and widens output dtype. |
| mul | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise multiply with broadcasting. Accumulate is integer-only and widens output dtype. |
| matmul | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | N-D matmul with batch-dim broadcasting. Accumulate is integer-only and widens output dtype. |
| abs | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise absolute value. Accumulate is integer-only and widens output dtype. |
| relu | [f8, bf16, f16, f32, f64]<br>[i4, i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] | negative_slope, clamp_max | Yes | No | Leaky ReLU with clamp. |
| is_finite | [f8, bf16, f16, f32, f64] | a: Tensor[T] | y: bool (scalar) | None | No | No | True if all elements are finite. |
| fill | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T] | y: Tensor[T] | value | Yes | No | Fill output with a scalar literal. |

Device notes:

- CPU is the reference implementation for all listed dtypes.
- AVX/AVX2 match CPU dtype coverage; i1/u1 packed types fall back to CPU.
- Vulkan supports all listed dtypes with shader-side casting for f8/bf16/f16.
- Vulkan add/mul/matmul support packed broadcast, including inplace/accumulate.
- Vulkan relu supports inplace execution.
- If Vulkan lacks `shader_int64` or `shader_float64`, ops fall back to CPU with a warning.

## CPU Fallback Coverage (Non-CPU Backends)

Fallbacks are per-op and per-dtype. Tables below list the dtypes that may fall
back to CPU when using the non-CPU backend.

### CPU AVX

| Op name | Dtypes that fall back to CPU |
| --- | --- |
| add | i1, u1 |
| mul | i1, u1 |
| matmul | i1, u1 |
| abs | i1 |
| relu | none |
| is_finite | none |
| fill | i1, u1 |

### CPU AVX2

| Op name | Dtypes that fall back to CPU |
| --- | --- |
| add | i1, u1 |
| mul | i1, u1 |
| matmul | i1, u1 |
| abs | i1 |
| relu | none |
| is_finite | none |
| fill | i1, u1 |

### Vulkan

These fallbacks are conditional on GPU feature support:

| Op name | Dtypes that fall back to CPU |
| --- | --- |
| add | i64, u64, f64 (if missing `shader_int64` / `shader_float64`) |
| mul | i64, u64, f64 (if missing `shader_int64` / `shader_float64`) |
| matmul | i64, u64, f64 (if missing `shader_int64` / `shader_float64`) |
| abs | i64, f64 (if missing `shader_int64` / `shader_float64`) |
| relu | i64, f64 (if missing `shader_int64` / `shader_float64`) |
| is_finite | f64 (if missing `shader_float64`) |
| fill | i64, u64, f64 (if missing `shader_int64` / `shader_float64`) |
