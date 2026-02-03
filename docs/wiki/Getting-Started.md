# Getting Started

This guide shows a minimal path to running OpenInfer locally.

## Prerequisites

- Rust toolchain (`cargo`)
- Python 3 + pip (for `.oinf` tooling and Python examples)
- Optional: Vulkan SDK + `slangc` (only if you want GPU execution)

Install Python deps:

```bash
pip install -r requirements.txt
```

## Build

CPU only:

```bash
cargo build -p openinfer
```

Vulkan (optional):

```bash
cargo build -p openinfer --features vulkan
```

Build SPIR‑V (optional, for Vulkan shaders):

```bash
cargo build-spv
```

## Generate a sample model

```bash
python examples/openinfer-oinf/mlp_regression_oinf.py
```

This writes `res/models/mlp_regression.oinf`.

## Run a Rust example

```bash
cargo run -p openinfer --example mlp_regression
```

Select a device (if Vulkan is enabled):

```bash
cargo run -p openinfer --example mlp_regression -- --target=cpu
cargo run -p openinfer --example mlp_regression --features vulkan -- --target=vulkan
```

## Next steps

- Learn the DSL and memory kinds: [Core Concepts](Core-Concepts)
- See what ops and dtypes are supported: [Capabilities](Capabilities)
- Get a high‑level view of the internals: [Architecture](Architecture)
- Learn how to run tests and set up aliases: [Testing and Tools](Testing-and-Tools)
