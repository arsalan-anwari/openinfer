# Getting Started

This guide shows a minimal path to running OpenInfer locally.

## Prerequisites

- Rust toolchain (`cargo`)
- Python 3 + pip (for `.oinf` tooling and Python examples)
- Optional: Vulkan SDK + `slangc` (only if you want GPU execution)

## User method (packages)

Install the packages directly (recommended for users):

Rust crates:
```bash
cargo add openinfer-simulator
cargo add openinfer-dsl
cargo add openinfer-synth
```

Python tooling:
```bash
pip install openinfer-oinf
```

## Repo method (contributors)

Enable recursive submodule updates once:
```bash
git config --global submodule.recurse true
git config --global fetch.recurseSubmodules true
```

Or run:
```bash
./scripts/bootstrap_submodules.sh
```

## Setup (recommended)

```bash
./scripts/setup_all.sh
./scripts/sync_models.sh
```

Install Python deps manually (if needed):

```bash
pip install -r requirements.txt
```

## Build

CPU only:

```bash
cargo build --manifest-path openinfer-simulator/Cargo.toml
```

Vulkan (optional):

```bash
cargo build --manifest-path openinfer-simulator/Cargo.toml --features vulkan
```

Build SPIR‑V (optional, for Vulkan shaders):

```bash
cargo build-spv
```

## Generate a sample model

```bash
python openinfer-oinf/examples/mlp_regression_oinf.py
```

This writes `openinfer-oinf/res/models/mlp_regression.oinf`.

## Run a Rust example

```bash
cargo run --manifest-path openinfer-simulator/Cargo.toml --example mlp_regression
```

Select a device (if Vulkan is enabled):

```bash
cargo run --manifest-path openinfer-simulator/Cargo.toml --example mlp_regression -- --target=cpu
cargo run --manifest-path openinfer-simulator/Cargo.toml --example mlp_regression --features vulkan -- --target=vulkan
```

## Next steps

- Learn the DSL and memory kinds: [Core Concepts](Core-Concepts)
- See what ops and dtypes are supported: [Capabilities](Capabilities)
- Get a high‑level view of the internals: [Architecture](Architecture)
- Learn how to run tests and set up aliases: [Testing and Tools](Testing-and-Tools)
