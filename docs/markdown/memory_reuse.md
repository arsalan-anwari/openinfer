# Memory Reuse and Trade-offs

This document explains the memory reuse decisions made in OpenInfer, why some
ops are slower than pure compute kernels, and how this keeps large models
practical on limited hardware.

## Why memory reuse matters

OpenInfer is designed to load and execute very large tensors (2GB+ per tensor).
When multiple large buffers are live at once, allocating new outputs for every
op can exhaust system RAM or GPU memory. If the machine has no swap space,
allocation failure can crash the process or the OS.

To keep models runnable:

- Accumulation ops can reuse a preallocated output buffer when available.
- Inplace execution is supported when outputs alias inputs.
- Packed and low-bit types are operated on in-place rather than expanded.

These choices reduce memory churn and peak allocations.

## Why some ops are slower

Memory conservation introduces work that is not present in a naive kernel:

- Packed integer ops must unpack/operate/repack per element.
- Low-bit floats are cast inline per element in shaders or kernels.
- Accumulation ops validate output compatibility and may copy less-contiguous data.
- In-place or reuse-friendly code paths can reduce SIMD-friendly layouts.

The simulator prioritizes correctness and the ability to load large weights over
raw per-op throughput. This is intentional for debugging, correctness checks,
and running large models on constrained systems.

## Practical guidance

- If you need maximum throughput, prefer GPU backends and ensure enough memory.
- If you need stability with very large tensors, enable reuse and avoid
  unnecessary intermediate allocations.
- When running large models on machines without swap, avoid launching multiple
  large runs concurrently.
