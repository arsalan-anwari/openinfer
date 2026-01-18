# Types

This page summarizes supported tensor element types and how they are handled on
different devices.

## Supported DTypes

Universal types (allowed everywhere, including scalars):

- f64
- f32
- i64
- i32
- i16
- i8
- u64
- u32
- u16
- u8
- bool

Special tensor-only types:

- f16
- bf16
- f8 (float8e5m2)
- i4
- i2
- i1

Packed integer types (i1/i2/i4) store multiple elements per byte in OINF blobs.
Packing is LSB-first within each byte.

## CPU Backend

- Supports all universal and special tensor types.
- f16/bf16/f8 are upsampled to f32 for kernel execution and converted back to
  the original dtype after execution.
- Packed types (i1/i2/i4) remain packed; kernels are responsible for unpacking
  and applying the math logic.

## Vulkan Backend

Hardware feature checks:

- i64/u64 require `shader_int64`.
- f64 requires `shader_float64`.
- f16 presence is detected via `shader_float16` but is still upsampled.

Upsampling behavior:

- f8/bf16/f16 are upsampled to f32 for Vulkan kernels.
- f8/bf16 are never assumed to be natively supported on Vulkan devices.
- f16 is also upsampled today for consistency.

Packed types:

- i1/i2/i4 are stored as packed bytes in GPU buffers.
- Kernels must unpack these values and implement the arithmetic in packed form.
- Packed types are never upsampled to i8.

## Errors vs Upsampling

- If a Vulkan device does not support `shader_float64`, any f64 usage in the DSL
  returns an error.
- If a Vulkan device does not support `shader_int64`, i64/u64 usage returns an
  error.
- Lower-bit float types (f8/bf16/f16) are always upsampled to f32.
