# Supported Ops

This document lists currently supported ops and their backend coverage.

## Backend overview

- **CPU:** scalar Rust kernels.
- **CPU (AVX/AVX2):** SIMD kernels for x86_64 when enabled.
- **Vulkan:** compute kernels compiled from Slang.

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

## Op coverage

The tables below list support by device. Each row includes input/output shapes,
attributes, dtype coverage (grouped by float/signed/unsigned/packed), and
device-specific notes.

---

## CPU

| Op name | Dtypes | Input | Output | Attributes | Inplace | Accumulation | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise add with broadcasting. Accumulate is integer-only. |
| mul | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise multiply with broadcasting. Accumulate is integer-only. |
| abs | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[i1, i2, i4] | a: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise absolute value. Accumulate is integer-only. |
| relu | [f8, bf16, f16, f32, f64]<br>[i4, i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] | negative_slope, clamp_max | No | No | Leaky ReLU with clamp. |
| is_finite | [f8, bf16, f16, f32, f64] | a: Tensor[T] | y: bool (scalar) | None | No | No | True if all elements are finite. |
| fill | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T] | y: Tensor[T] | value | Yes | No | Fill output with a scalar literal. |

CPU notes:

- Scalar kernels cover all listed dtypes, including packed types via bit-level ops.

---

## CPU (AVX)

| Op name | Dtypes | Input | Output | Attributes | Inplace | Accumulation | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | [f32, f64]<br>[i8, i16]<br>[u8, u16] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | SIMD-only add. Accumulate limited to i8→i16, i16→i32, i32→i64, u8→u16, u16→u32, u32→u64. |
| mul | [f32, f64]<br>[i8, i16, i32]<br>[u8, u16, u32] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | SIMD-only multiply. Accumulate limited to i8→i16 and u8→u16. |
| abs | [f32, f64]<br>[i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] |  | Yes | Yes | SIMD-only abs. Accumulate limited to i8→i16 and i16→i32. |
| relu | [f32] | a: Tensor[T] | y: Tensor[T] | negative_slope, clamp_max | No | No | SIMD-only relu (f32 only). |
| fill | - | a: Tensor[T] | y: Tensor[T] | value | No | No | Unsupported on AVX. |
| is_finite | - | a: Tensor[T] | y: bool (scalar) | None | No | No | Unsupported on AVX. |
| matmul | - | a: Tensor[T], b: Tensor[T] | y: Tensor[T] | None | No | No | Unsupported on AVX. |

AVX notes:

- SIMD-only: packed, bool/bitset, and f8/bf16/f16 are unsupported.
- Any dtype not covered above returns “unsupported op on this device”.

---

## CPU (AVX2)

| Op name | Dtypes | Input | Output | Attributes | Inplace | Accumulation | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | [f32, f64]<br>[i8, i16]<br>[u8, u16] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | SIMD-only add. Accumulate limited to i8→i16, i16→i32, i32→i64, u8→u16, u16→u32, u32→u64. |
| mul | [f32, f64]<br>[i8, i16, i32]<br>[u8, u16, u32] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | SIMD-only multiply. Accumulate limited to i8→i16 and u8→u16. |
| abs | [f32, f64]<br>[i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] |  | Yes | Yes | SIMD-only abs. Accumulate limited to i8→i16 and i16→i32. |
| relu | [f32] | a: Tensor[T] | y: Tensor[T] | negative_slope, clamp_max | No | No | SIMD-only relu (f32 only). |
| fill | - | a: Tensor[T] | y: Tensor[T] | value | No | No | Unsupported on AVX2. |
| is_finite | - | a: Tensor[T] | y: bool (scalar) | None | No | No | Unsupported on AVX2. |
| matmul | - | a: Tensor[T], b: Tensor[T] | y: Tensor[T] | None | No | No | Unsupported on AVX2. |

AVX2 notes:

- SIMD-only: packed, bool/bitset, and f8/bf16/f16 are unsupported.
- Any dtype not covered above returns “unsupported op on this device”.

---

## Vulkan

| Op name | Dtypes | Input | Output | Attributes | Inplace | Accumulation | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise add with broadcasting. Packed dtypes decode/encode in shader. |
| mul | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise multiply with broadcasting. Packed dtypes decode/encode in shader. |
| abs | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[i1, i2, i4] | a: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Elementwise abs. Packed dtypes decode/encode in shader. |
| relu | [f8, bf16, f16, f32, f64]<br>[i4, i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] | negative_slope, clamp_max | No | No | Leaky ReLU with clamp. |
| matmul | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T] (M×K), b: Tensor[T] (K×N) | y: Tensor[T] (M×N) |  | Yes | Yes | Matmul. Packed dtypes decode/encode in shader. |
| is_finite | [f8, bf16, f16, f32, f64] | a: Tensor[T] | y: bool (scalar) | None | No | No | True if all elements are finite. |
| fill | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[i1, i2, i4, u1, u2, u4]<br>[bool, bitset] | a: Tensor[T] | y: Tensor[T] | value | Yes | No | Fill output with a scalar literal. Packed dtypes decode/encode in shader. |

---

Vulkan notes:

- Packed types are processed in-shader using byte-addressed buffers; no scalar expansion.
- f16/bf16/f8 are upcast to f32 buffers on upload and converted back on download.
- i64/u64 require `shader_int64` and f64 requires `shader_float64`; unsupported GPUs error.
- `bitset` is treated as an 8-bit scalar in Vulkan shaders.