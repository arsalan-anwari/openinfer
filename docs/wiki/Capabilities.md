# Capabilities and Support

This page summarizes what OpenInfer supports today at a user level.

## Ops

The current op catalog is defined in `ops.json` and includes:

- Arithmetic: `add`, `mul`, `sub`, `div`, `floor_div`, `rem`, `fma`, `neg`, `recip`
- Math/activations: `abs`, `relu`, `sign`, `clamp`, `floor`, `ceil`, `round`, `trunc`
- Matrix: `matmul`
- Bitwise: `and`, `or`, `xor`, `not`, `shl`, `shr`, `popcount`
- Comparison: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Filtering: `is_finite`, `is_nan`, `is_inf`, `is_neg`, `filter`
- Reductions: `sum_axis`, `mean_axis`, `prod_axis`, `max_axis`, `min_axis`,
  `argmax_axis`, `argmin_axis`
- Casting: `cast`

For per‑op dtype coverage and flags, see [Ops and DType Support](Ops) or
`ops.json` / `res/ops_v2.json`.

## DTypes

Supported types include:

- Floats: `f8`, `bf16`, `f16`, `f32`, `f64`
- Signed ints: `i1`, `i2`, `i4`, `i8`, `i16`, `i32`, `i64`
- Unsigned ints: `u1`, `u2`, `u4`, `u8`, `u16`, `u32`, `u64`
- `bool`, `bitset`

Packed integer types are stored densely in `.oinf` files and remain packed in
execution.

## Backends

- **CPU**: Always available and used for validation
- **Vulkan**: Optional, feature‑gated (`--features vulkan`)

If a Vulkan device lacks `shader_int64` or `shader_float64`, those dtypes fall
back to CPU with a warning.

## Numerical drift

CPU and GPU results can differ slightly due to low‑precision formats and GPU
execution order. This is expected, especially for `f8`, `bf16`, and `f16`.

## Synthesis (planned)

OpenInfer does not yet ship a synthesizer. A future pipeline is planned to lower
graphs into optimized, backend‑specific code for targets like CPU, GPU, TPU, and
FPGA, including vendor‑specific optimization passes.
