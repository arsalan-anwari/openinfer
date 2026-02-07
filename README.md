# Open Infer

![](res/images/OpenInferLogo.png)

_Inference graphs, explicit control flow, portable execution._

`inference` · `dsl` · `graph` · `cpu` · `vulkan` · `ml`

OpenInfer is an **edge-focused ML transpilation framework**. It lets developers
express inference pipelines in a Rust-embedded DSL, validate them in a
high-level simulator, and then synthesize fully static, device-specific source
code for deployment on constrained hardware.

The simulator runs purely on the host and verifies graph correctness,
scheduling logic, memory layouts, and transformations inside the DSL. A planned
synthesizer lowers the same graph into concrete C/CUDA/Vulkan/VHDL-style source
that targets a thin hardware-abstraction layer and native device APIs.

---

## Highlights

- ✨ **Explicit graphs** with visible control flow and side effects
- 🧩 **Model‑agnostic**: transformers, vision, audio, streaming pipelines
- 🔍 **Inspectable**: tracing, timing, and JSON serialization
- 🧠 **Host simulator** for correctness, scheduling, and memory layout checks
- ⚡ **Synthesizer (planned)**: lower graphs into static, device-specific source

## Overview

OpenInfer defines a symbolic, inspectable inference graph that can be simulated,
traced, and validated on the host before being lowered into device-specific
source code. It is designed for edge inference, custom accelerators, FPGA
pipelines, and safety-critical firmware where teams want transparent generated
code and strict control over memory and scheduling.

The main website is [www.open-infer.nl](https://www.open-infer.nl), and the docs
live at [docs.open-infer.nl](https://docs.open-infer.nl).

### Condensed Rust Example (DSL Overview)

```rust
use openinfer::{
    fetch_executor, graph, insert_executor, Device, ModelLoader, Random, Simulator, Tensor,
    TensorOptions,
};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("openinfer-simulator/res/models/mlp_regression.oinf")?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            alpha: f32;
            num_layers: u32;
        }

        volatile {
            W(l): f32[D, D] @pattern("W.{l}");
            y: f32[B, D];
        }

        persistent {
            step: i32 @init(0);
            K(l, t): f32[B, D] @table;
        }

        block entry {
            assign h: f32[B, D];
            assign cond: bool;

            op matmul(x, W[0]) >> h;
            barrier;

            loop layers (l in 0..num_layers) {
                cache.read K[l, step] >> h;
                op relu(h, alpha=0.01, clamp_max=6.0) >> h;
                dep after(relu) before(cache.write);
                cache.write h >> K[l, step];
            }

            cache.increment step;
            op is_finite(h) >> cond;
            branch cond ok bad;
            return;
        }

        block ok {
            op add(h, alpha) >> y;
            return;
        }

        block bad {
            op fill(y, value=0.0) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, Device::Cpu)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let input = Random::<f32>::generate_with_seed_opts(
        0,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });

    Ok(())
}
```
> This snippet is illustrative. Your `.oinf` model should define matching
> sizevars and tensors for the variables referenced in the graph.
> See the Wiki for current [op support](../../wiki/Ops) and [capabilities](../../wiki/Capabilities).

## Documentation (Docs + Wiki)

- Docs homepage: [docs.open-infer.nl](https://docs.open-infer.nl)
- Main website: [www.open-infer.nl](https://www.open-infer.nl)
- Wiki home and quick links: [Wiki Home](../../wiki)
- Getting started: [Getting Started](../../wiki/Getting-Started)
- Core concepts and DSL: [Core Concepts](../../wiki/Core-Concepts)
- Components and usage: [Components](../../wiki/Components), [Using OpenInfer](../../wiki/Using-OpenInfer)
- Ops + dtype support: [Ops](../../wiki/Ops)
- Capabilities + roadmap: [Capabilities](../../wiki/Capabilities), [Synthesis](../../wiki/Synthesis)
- Testing + tools: [Testing and Tools](../../wiki/Testing-and-Tools)

## Modules

- `openinfer-simulator`: host-side simulator and runtime (`openinfer-simulator/`)
- `openinfer-dsl`: Rust-embedded DSL for graph construction
- `openinfer-oinf`: Python tooling for the `.oinf` model format
- `openinfer-synth`: synthesis pipeline for device-specific codegen

## User method (packages)

Install from package managers (recommended for users):

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

If you are contributing to the repo, use submodules.

To ensure submodules are fetched automatically on clone/pull, set this once:
```bash
git config --global submodule.recurse true
git config --global fetch.recurseSubmodules true
```

If you cannot set global config, use:
```bash
./scripts/bootstrap_submodules.sh
```

To update submodules later:
```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Quickstart (recommended)

Prerequisites:
- Rust toolchain (cargo)
- Python 3 + pip (for OINF tools and Python examples)
- Slang compiler (`slangc`) for Vulkan shader builds (add to PATH)

Setup + build + tests:
```bash
./scripts/setup_all.sh
./scripts/sync_models.sh
./scripts/build_all.sh
./scripts/run_tests.sh
```

For Vulkan builds with shader progress output, use:
```bash
cargo build-spv
```

To clean Vulkan SPIR-V artifacts and then run `cargo clean`:
```bash
cargo clean-spv
```

## Run Examples

Using the runner script (recommended):
```bash
./scripts/run_examples.sh
```

Targeted modes:
```bash
./scripts/run_examples.sh --target cpu
./scripts/run_examples.sh --target vulkan --features vulkan
./scripts/run_examples.sh --target all --features vulkan
```

Manual (per-submodule):
```bash
python openinfer-oinf/examples/{example}_oinf.py
python openinfer-oinf/verify_oinf.py "openinfer-oinf/res/models/{example}.oinf"
cargo run --manifest-path openinfer-simulator/Cargo.toml --example {name}
```

## Tests

Run the full test suite:
```bash
./scripts/run_tests.sh
```

Common options:
```bash
./scripts/run_tests.sh --list
./scripts/run_tests.sh --target=cpu
./scripts/run_tests.sh --target=vulkan --features=vulkan
./scripts/run_tests.sh --target=all --features=vulkan
./scripts/run_tests.sh --test-filter openinfer-simulator::ops_misc
```

## Supported Targets

- Simulator: CPU host execution with optional Vulkan backend
- Planned synthesis targets: ARM + NEON, x86 + AVX, Vulkan GPUs, NVIDIA Jetson
  CUDA, Android NNAPI, USB TPUs (Coral), bare-metal MCUs, and FPGA flows
  (VHDL/HLS)

## Synthesizer (Planned)

OpenInfer does not yet ship a synthesizer, but the long-term goal is to lower
DSL graphs into optimized, backend-specific code (C/CUDA/Vulkan/VHDL/HLS). This
enables native output for edge targets while keeping compilation and deployment
inside vendor toolchains.

## Status

OpenInfer is in early development. See [Capabilities](../../wiki/Capabilities)
for current coverage and roadmap notes.

---


## License

Apache-2.0
