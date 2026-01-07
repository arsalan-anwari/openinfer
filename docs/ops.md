Supported Ops
=============

This document lists currently supported ops and their backend coverage.

Backend Overview
----------------
- CPU: scalar Rust kernels.
- CPU (AVX/AVX2): SIMD kernels for x86_64 when enabled.
- Vulkan: compute kernels compiled from Slang.

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
