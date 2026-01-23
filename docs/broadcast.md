# Broadcasting

Broadcasting is opt-in per op. When enabled, inputs with different shapes are
expanded to a common shape before the op runs. Op kernels still receive flat
slices and do not need shape-aware logic.

## Policy and Where to Configure It

Set the policy in `openinfer/src/ops/registry.rs`:

- `BroadcastPolicy::None`: broadcasting disabled (default)
- `BroadcastPolicy::CpuOnly`: elementwise broadcast on CPU, require identical shapes on Vulkan
- `BroadcastPolicy::AllDevices`: elementwise broadcast on CPU and Vulkan
- `BroadcastPolicy::BatchOnly`: batch-dim broadcast for matmul-like ops (all devices)

The switch lives in `broadcast_policy(op: OpKind)`.

## CPU Behavior

When broadcasting is enabled for an op, the CPU backend:

1) Computes the output shape with `broadcast_shapes`
2) For `add`/`mul`, uses stride-aware indexing to read inputs without materializing
3) Runs the op kernel on the broadcasted shape

If broadcasting is disabled, mismatched shapes return an error.

## Vulkan Behavior

When broadcasting is enabled for an op, the Vulkan backend:

1) Computes the output shape with `broadcast_shapes`
2) For `add`/`mul`, passes per-input metadata and indexes inputs directly in the shader
3) Runs the op kernel without allocating expanded buffers

Broadcast expansion also applies to inplace variants on Vulkan when the op
supports broadcasting, so `op add(x, y) >> x` can work with broadcasted `y`.

The broadcast pass uses `openinfer/src/ops/vulkan/broadcast/broadcast.slang`.

Timer notes:
- The broadcast pass does not write to the timer.
- Op durations only cover the actual op kernel dispatch.

## Limits and Notes

- Broadcasting only applies to multi-input ops.
- Packed dtypes (`i1`, `i2`, `i4`, `u1`, `u2`, `u4`) support broadcast on CPU and Vulkan for add/mul/matmul, including inplace and accumulate variants.
- Batch broadcast for `matmul` is supported for packed and non-packed dtypes (including inplace and accumulate variants), with packed offsets requiring byte-aligned batch strides.
- Vulkan broadcast metadata is encoded as `u32`, so shapes and strides must fit
  in `u32` values.
