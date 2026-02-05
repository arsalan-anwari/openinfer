# Accumulation (acc=) Rules

This document describes the accumulation (`acc=`) attribute and how it is used.

## Summary

- `acc=` enables accumulation into a wider integer output type to reduce overflow.
- It is opt-in and only available on specific ops (example: `add`, `mul`, `abs`, `matmul`).
- The output tensor dtype must match the `acc=` type.
- `acc=` disables aliasing the output with an input, but it can reuse a preallocated
  output buffer when the DSL output already exists and matches type/shape.

## Supported Types

Accumulation is **integer-only**:

- **Signed integers:** `i1`, `i2`, `i4`, `i8`, `i16`, `i32`, `i64`
- **Unsigned integers:** `u1`, `u2`, `u4`, `u8`, `u16`, `u32`, `u64`

Not supported with `acc=`:

- Floats: `f8`, `bf16`, `f16`, `f32`, `f64`
- Non-numeric: `bool`, `bitset`
- Ternary packed: `t1`, `t2`

## Valid Accumulate Pairs

For `acc=` to be valid:

- Inputs must be **the same dtype**.
- `acc=` must have **the same signedness** as inputs.
- `acc=` must be **wider** than the input dtype.

Examples:

- `i8 -> i16`, `i8 -> i32`, `i8 -> i64`
- `i16 -> i32`, `i16 -> i64`
- `i32 -> i64`
- `u8 -> u16`, `u8 -> u32`, `u8 -> u64`
- `u16 -> u32`, `u16 -> u64`
- `u32 -> u64`
- Packed inputs widen to scalar outputs, e.g. `i4 -> i8`, `u2 -> u16`

Invalid examples:

- `i8 -> i8` (not wider)
- `u8 -> i16` (signedness mismatch)
- `f16 -> f32` (floats not allowed)

## Output Buffer Reuse

If the output tensor already exists (same dtype and shape), the executor passes
it into the accumulation kernel so the op can reuse the buffer instead of
allocating a fresh one. This is important when tensors are large or memory is
tight. When no compatible output exists, the kernel allocates a new buffer.

## Graph Validation

The simulator validates accumulation usage:

- Ensures `acc=` is only on supported ops.
- Ensures output dtype matches `acc=`.
- Ensures `acc=` is wider and same signedness as inputs.
- Rejects in-place aliasing with inputs when `acc=` is present.
