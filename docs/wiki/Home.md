# OpenInfer Wiki

![](../../res/images/OpenInferHeader.png)

OpenInfer is an inference graph and execution framework for machine-learning
workloads. It lets you describe inference logic and control flow explicitly in
Rust, then simulate and execute the graph on CPU or Vulkan.

Use this wiki for a user‑level overview: what OpenInfer does, what components it
has, and how to use it. Detailed contributor and implementation docs live
elsewhere in the repo.

## Quick links

- [Getting Started](Getting-Started)
- [Using OpenInfer](Using-OpenInfer)
- [Core Concepts](Core-Concepts)
- [Components](Components)
- [Capabilities and Support](Capabilities)
- [Ops and DType Support](Ops)
- [Architecture (Overview)](Architecture)
- [Testing and Tools](Testing-and-Tools)
- [Synthesis Roadmap](Synthesis)
- [FAQ](FAQ)

## What OpenInfer is

- **Explicit graphs**: control flow is visible and deterministic
- **Model‑agnostic**: works for transformer‑like, vision, or streaming pipelines
- **Inspectable**: supports tracing, serialization, and reproducible runs
- **Portable**: CPU backend + Vulkan backend (feature‑gated)

## What you can do

- Load a `.oinf` model package (weights + metadata + sizevars)
- Build a graph with the Rust DSL
- Simulate and run on CPU or Vulkan
- Fetch results and traces for inspection

## Where to look next

- Try the [Getting Started](Getting-Started) flow
- Browse [Core Concepts](Core-Concepts) to understand memory kinds and control flow
- Review [Capabilities and Support](Capabilities) for ops/dtypes and backend notes
