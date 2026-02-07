# OpenInfer Wiki

![](https://github.com/arsalan-awnari/openinfer/blob/main/res/images/OpenInferHeader.png)

OpenInfer is an edge-focused ML transpilation framework. It lets developers
express inference pipelines in a Rust-embedded DSL, validate them in a
high-level simulator, and then synthesize fully static, device-specific source
code for deployment on constrained hardware.

The simulator runs purely on the host to verify graph correctness, scheduling,
memory layouts, and DSL transformations. A planned synthesizer lowers the same
graph into concrete C/CUDA/Vulkan/VHDL-style source that targets a thin
hardware abstraction layer and native device APIs.

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
- **Model‑agnostic**: works for transformer-like, vision, or streaming pipelines
- **Inspectable**: supports tracing, serialization, and reproducible runs
- **Edge-focused**: simulator + synthesis workflow for constrained targets

## What you can do

- Load a `.oinf` model package (weights + metadata + sizevars)
- Build a graph with the Rust DSL
- Validate and simulate on the host
- Prepare for synthesis into device-specific code

## User method (packages)

Use package managers to install OpenInfer components:

- Rust crates: `openinfer-simulator`, `openinfer-dsl`, `openinfer-synth`
- Python tooling: `pip install openinfer-oinf`

## Where to look next

- Try the [Getting Started](Getting-Started) flow
- Browse [Core Concepts](Core-Concepts) to understand memory kinds and control flow
- Review [Capabilities and Support](Capabilities) for ops/dtypes and backend notes
