# GPU / CPU Float Drift

Floating-point results can differ slightly between CPU and GPU implementations.
This is expected even when both paths are correct, especially for low-bit
formats (f8/bf16/f16).

## Why drift happens

- Different rounding rules or evaluation order (e.g. fused multiply-add vs
  separate multiply and add).
- Low-bit formats require conversion to/from f32, which can round differently
  across implementations.
- Shader compilers may reorder or combine operations unless explicitly constrained.
- Accumulation order can differ in parallel workloads.

The Vulkan backend always casts f8/bf16 to f32 inside shaders and writes back to the
original dtype. For f16, shaders use native half when `shader_float16` is available
and fall back to f32 casting otherwise. Even with careful rounding, perfectly
matching CPU is not always possible across all GPUs and drivers.

## Drift tolerance used in validation

The reference validation in `examples/rust/ops_accumulate_inplace.rs` uses the
following tolerances. Values within tolerance are marked with `⚠️`:

- f16: abs 0.6, rel 0.08
- bf16: abs 0.1, rel 0.02
- f8: abs 0.6, rel 0.25
- f32: abs 1e-4, rel 1e-4
- f64: abs 1e-8, rel 1e-8

These thresholds reflect typical drift observed across devices while still
flagging real numerical errors.

## Interpreting results

- `✅`: exact match with the CPU reference.
- `⚠️`: within tolerance (drift, but acceptable).
- `❌`: outside tolerance (likely a real bug).
