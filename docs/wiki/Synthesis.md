# Synthesis Roadmap

OpenInfer today runs graphs via the simulator/executor on CPU or Vulkan. A
future direction is a **synthesizer** that lowers the DSL graph into optimized,
backend‑specific code.

## What this would enable (planned)

- Generate native source for multiple targets (CPU/GPU/TPU/FPGA).
- Apply backend‑specific scheduling, memory planning, and fusion.
- Allow hardware‑specific optimization passes for different vendors.

## Status

This synthesizer is **not implemented yet**. The current runtime focuses on
correctness, tracing, and explicit control flow. The synthesis pipeline is a
planned capability and will evolve alongside device‑architecture descriptions
and optimization passes.

## Why it matters

The goal is to keep the DSL expressive while enabling high‑performance backends
without rewriting model logic. For example, the same graph could be lowered to
an embedded GPU or a system‑on‑chip target with tailored optimizations.
