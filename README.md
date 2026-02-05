# Open Infer

![](res/images/OpenInferLogo.png)

_Inference graphs, explicit control flow, portable execution._

`inference` · `dsl` · `graph` · `cpu` · `vulkan` · `ml`

OpenInfer is an open-source **inference graph and execution framework** for
machine-learning workloads. It lets you describe inference logic and control
flow explicitly in a Rust‑embedded DSL, then simulate and execute on CPU or
Vulkan. A key future pillar is the **synthesizer** (planned), which will lower
DSL graphs into optimized, backend‑specific code for targets like CPU, GPU, TPU,
and FPGA.

The focus is **clarity, explicit control, and inspectability**, rather than
hiding complexity behind opaque runtimes.

---

## Highlights

- ✨ **Explicit graphs** with visible control flow and side effects
- 🧩 **Model‑agnostic**: transformers, vision, audio, streaming pipelines
- 🔍 **Inspectable**: tracing, timing, and JSON serialization
- ⚡ **Portable**: CPU backend + optional Vulkan backend
- 🧠 **Synthesizer (planned)**: lower DSL graphs to optimized backend code

## Overview

OpenInfer defines a symbolic, inspectable inference graph that can be simulated,
traced, and executed on CPU or Vulkan. The long‑term plan is a synthesizer that
generates optimized, backend‑specific code from the same graph. The main website
is [www.open-infer.nl](https://www.open-infer.nl), and the docs live at
[docs.open-infer.nl](https://docs.open-infer.nl).

### Condensed Rust Example (DSL Overview)

```rust
use openinfer::{
    fetch_executor, graph, insert_executor, Device, ModelLoader, Random, Simulator, Tensor,
    TensorOptions,
};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("res/models/mlp_regression.oinf")?;

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

## Prerequisites

- Rust toolchain (cargo)
- Python 3 + pip (for OINF tools and Python examples)
- Slang compiler (`slangc`) for Vulkan shader builds (add to PATH)
- Python deps: `pip install -r requirements.txt`

## Build

```bash
cargo build -p openinfer
cargo build -p openinfer --features vulkan
```

For Vulkan builds with shader progress output, use:
```bash
cargo build-spv
```

To clean Vulkan SPIR-V artifacts and then run `cargo clean`:
```bash
cargo clean-spv -p openinfer
```

## Run Examples

### Python
```bash
python examples/openinfer-oinf/{example}_oinf.py
python openinfer-oinf/verify_oinf.py "res/models/{example}.oinf"
```

### Rust
```bash
cargo run -p openinfer --example {name}
```

Using the runner script (default, all examples):
```bash
./scripts/run_rust_examples.sh
```

Targeted modes:
```bash
./scripts/run_rust_examples.sh --target cpu
./scripts/run_rust_examples.sh --target vulkan --features vulkan
./scripts/run_rust_examples.sh --target all --features vulkan
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
./scripts/run_tests.sh --test-filter openinfer::ops_misc
```

## Supported Targets

- Architectures: CPU (scalar kernels) and Vulkan GPU backend
- SIMD extensions: AVX/AVX2 are enabled via `.cargo/config.toml` on x86_64 Linux
- GPU drivers: Vulkan-capable drivers (feature-gated)
- Vulkan dtype support: see [Capabilities](../../wiki/Capabilities); f16 is always simulated via f32 casts, and f64/i64/u64 depend on device features (fallback to CPU when unsupported)

## Synthesizer (Planned)

OpenInfer does not yet ship a synthesizer, but the long‑term goal is to lower
DSL graphs into optimized, backend‑specific code. This would enable native
output for targets like CPU, GPU, TPU, and FPGA, with room for vendor‑specific
optimization passes as the ecosystem matures.

## Status

OpenInfer is in early development. See [Capabilities](../../wiki/Capabilities)
for current coverage and roadmap notes.

---


## License

Apache-2.0
