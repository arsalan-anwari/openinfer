# Supported Ops

OpenInfer’s op definitions are sourced from `ops.json` and loaded by
`openinfer/src/op_defs.rs`. The JSON captures:

- Op names, arity, and attributes
- Broadcast/in-place/accumulation support
- Per-op dtype coverage (CPU + Vulkan)
- Output type rules

For a generated, tool-friendly view, see `res/ops_v2.json`.

## Current op set

This list is extracted from `ops.json`:

- `add`, `mul`, `sub`, `div`, `floor_div`, `rem`, `fma`, `neg`, `recip`
- `abs`, `relu`, `sign`, `clamp`, `floor`, `ceil`, `round`, `trunc`
- `matmul`
- `and`, `or`, `xor`, `not`, `shl`, `shr`, `popcount`
- `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- `is_finite`, `is_nan`, `is_inf`, `is_neg`, `filter`
- `sum_axis`, `mean_axis`, `prod_axis`, `max_axis`, `min_axis`, `argmax_axis`,
  `argmin_axis`
- `cast`

Use `ops.json` for the authoritative dtype matrix and per-op flags.
