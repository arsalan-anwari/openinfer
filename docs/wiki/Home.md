# OpenInfer Wiki

![](https://github.com/arsalan-awnari/openinfer/blob/main/res/images/OpenInferHeader.png)

OpenInfer is an edge-focused ML transpilation framework. It lets you describe
inference logic and control flow explicitly in Rust, validate graphs in a
host-side simulator, and then synthesize static, device-specific source code
for constrained hardware.

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

## Where to look next

- Try the [Getting Started](Getting-Started) flow
- Browse [Core Concepts](Core-Concepts) to understand memory kinds and control flow
- Review [Capabilities and Support](Capabilities) for ops/dtypes and backend notes
