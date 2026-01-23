# Types

This page summarizes supported tensor element types and how they are handled on
different devices.

## Supported DTypes

Universal types (allowed everywhere, including scalars):

- f32 / f64
- i8 / i16 / i32 / i64
- u8 / u16 / u32 / u64
- bool

Special tensor-only types:

- f8 (float8e5m2) / f16 / bf16
- i1 / i2 / i4
- u1 / u2 / u4
- t1 / t2 (reserved; not currently implemented in backends)

Packed integer types (i1/i2/i4/u1/u2/u4/t1/t2) store multiple elements per byte
in OINF blobs. Packing is LSB-first within each byte.

Packed dtype ranges:

- i1: {-1, 0}
- i2: {-2, -1, 0, 1}
- i4: {-8 ... 7}
- u1: {0, 1}
- u2: {0, 1, 2, 3}
- u4: {0 ... 15}
- t1: {-1, 1}
- t2: {-1, 0, 1}

## CPU Backend

- Supports all universal and special tensor types (t1/t2 reserved).
- f16/bf16/f8 are cast inline to f32 for arithmetic and converted back per element.
- Packed types remain packed; kernels read/write packed bytes directly without
  materializing full unpacked buffers.

## Vulkan Backend

Hardware feature checks and fallback:

- i64/u64 require `shader_int64`.
- f64 requires `shader_float64`.
- When these features are missing, Vulkan ops fall back to CPU with a warning
  (`eprintln!`) so execution continues instead of erroring.

Inline float handling:

- f8/bf16 are cast to f32 inside shaders per element (no intermediate f32 buffers
  on the host).
- f16 uses native half when `shader_float16` is available; otherwise shaders cast
  to f32 and write back. Use `Simulator::with_simulated_float()` to force the
  simulated f16 path for benchmarks or drift analysis.

Packed types:

- i1/i2/i4/u1/u2/u4 are stored as packed bytes in GPU buffers.
- Shaders decode/operate/encode in-place with byte-addressed buffers.
- Packed types are never expanded to i8 buffers.
## CPU AVX / AVX2 Backends

- Match CPU dtype coverage, including packed types.
- SIMD kernels operate directly on packed bytes for i2/i4/u2/u4.
- i1/u1 packed types fall back to the CPU implementation.
