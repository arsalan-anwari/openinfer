# Broadcasting

Broadcasting is opt-in per op. When enabled, inputs with different shapes are
expanded to a common shape before the op runs. Op kernels still receive flat
slices and do not need shape-aware logic.

## Policy and Where to Configure It

Set the policy in `openinfer/src/ops/registry.rs`:

- `BroadcastPolicy::None`: broadcasting disabled (default)
- `BroadcastPolicy::CpuOnly`: broadcast on CPU, require identical shapes on Vulkan
- `BroadcastPolicy::AllDevices`: broadcast on CPU and Vulkan

The switch lives in `broadcast_policy(op: OpKind)`.

## CPU Behavior

When broadcasting is enabled for an op, the CPU backend:

1) Computes the output shape with `broadcast_shapes`
2) Expands each input to the output shape
3) Runs the op kernel on equal-length slices

If broadcasting is disabled, mismatched shapes return an error.

## Vulkan Behavior

When broadcasting is enabled for an op, the Vulkan backend:

1) Computes the output shape with `broadcast_shapes`
2) Runs a GPU broadcast pass to expand each input buffer
3) Runs the op kernel on equal-length buffers

Broadcast expansion also applies to inplace variants on Vulkan when the op
supports broadcasting, so `op add(x, y) >> x` can work with broadcasted `y`.

The broadcast pass uses `openinfer/src/ops/vulkan/broadcast/broadcast.slang`.

Timer notes:
- The broadcast pass does not write to the timer.
- Op durations only cover the actual op kernel dispatch.

## Limits and Notes

- Broadcasting only applies to multi-input ops.
- Vulkan broadcast metadata is encoded as `u32`, so shapes and strides must fit
  in `u32` values.
