Supported Ops
=============

This document lists currently supported ops and their backend coverage.

Backend Overview
----------------
- CPU: scalar Rust kernels.
- CPU (AVX/AVX2): SIMD kernels for x86_64 when enabled.
- Vulkan: compute kernels compiled from Slang.

Adding Custom Ops
-----------------
This is the minimal checklist to add a new op end-to-end.

CPU
~~~
1) Add the op to `openinfer/src/graph.rs`:
   - `OpKind` variant and `as_str()` mapping.
   - Add any new attributes to `OpAttrs` if needed.
2) Add a CPU implementation under `openinfer/src/ops/cpu/<op>/`:
   - `mod.rs`, `<op>.rs`, `registry.rs`.
   - Optional inplace kernels: `<op>_inplace.rs` + `registry_inplace.rs`.
3) Register the op in the CPU registry:
   - `openinfer/src/ops/cpu/registry.rs` for out-of-place.
   - `openinfer/src/ops/cpu/registry_inplace.rs` via the op module.
4) If the op supports broadcasting, update the policy in
   `openinfer/src/ops/registry.rs` (`broadcast_policy`).

Vulkan
~~~~~~
1) Add the op under `openinfer/src/ops/vulkan/<op>/`:
   - `mod.rs`, `registry.rs`, and `<op>.slang`.
2) Add an entry to `openinfer/src/ops/vulkan/shaders.json` so `build.rs`
   compiles and embeds the SPIR-V. (Broadcast is a backend-only exception.)
3) Implement target selection in `openinfer/src/ops/vulkan/<op>/mod.rs`
   and add it to `openinfer/src/ops/vulkan/mod.rs` dispatch.
4) Register the Vulkan kernel in `openinfer/src/ops/vulkan/<op>/registry.rs`.
5) If you add an inplace variant, wire it in
   `openinfer/src/ops/vulkan/<op>/registry_inplace.rs`.
6) Ensure dtype constraints are enforced in the Vulkan backend
   (`openinfer/src/backend/vulkan/mod.rs`).

Op Coverage
-----------
Abs:
- CPU: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- CPU (AVX): i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- CPU (AVX2): i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- Vulkan: i8, i16, i32, i64, f32 (unsigned/bool are identity when enabled)

Add:
- CPU: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- CPU (AVX): i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- CPU (AVX2): i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- Vulkan: i8, i16, i32, i64, u8, u16, u32, u64, f32, bool

Mul:
- CPU: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- CPU (AVX): i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- CPU (AVX2): i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
- Vulkan: i8, i16, i32, i64, u8, u16, u32, u64, f32, bool

Notes
-----
- Vulkan dtype support is also constrained by `openinfer/src/backend/vulkan/mod.rs`.
- If an op lists a dtype but the GPU lacks the required feature, Vulkan will
  return an error at runtime.
