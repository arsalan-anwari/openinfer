```
  ░██████                                       
 ░██   ░██                                      
░██     ░██ ░████████   ░███████  ░████████     
░██     ░██ ░██    ░██ ░██    ░██ ░██    ░██    
░██     ░██ ░██    ░██ ░█████████ ░██    ░██    
 ░██   ░██  ░███   ░██ ░██        ░██    ░██    
  ░██████   ░██░█████   ░███████  ░██    ░██    
            ░██                                 
            ░██                                 
                                                
░██████               ░████                     
  ░██                ░██                        
  ░██  ░████████  ░████████  ░███████  ░██░████ 
  ░██  ░██    ░██    ░██    ░██    ░██ ░███     
  ░██  ░██    ░██    ░██    ░█████████ ░██      
  ░██  ░██    ░██    ░██    ░██        ░██      
░██████░██    ░██    ░██     ░███████  ░██      
                                                
                                                
                                                
```

# OpenInfer

OpenInfer is an open-source **inference graph and compilation framework** for machine-learning workloads.

Its primary goal is to let **developers describe inference logic and control flow explicitly**, using a clear, verbose, Rust-embedded DSL, while OpenInfer handles simulation, analysis, optimization, and code generation for specific hardware targets.

OpenInfer is **model-agnostic**. The same DSL can describe inference for:

* Transformers and LLMs
* CNNs and vision models
* Audio and signal-processing pipelines
* Streaming or recurrent architectures
* Experimental or custom ML systems

The focus is on **clarity, explicit control, and inspectability**, rather than hiding complexity behind opaque runtimes.

---

## Overview

OpenInfer defines a symbolic, inspectable inference graph that can be simulated, analyzed, and compiled into device-specific source code. The core workflow and mental model are captured in [docs/overview.md](docs/overview.md).

### Condensed Rust Example (All Components)

```rust
use openinfer::{
    cache, fetch_executor, graph, insert_executor, GraphDeserialize, GraphSerialize,
    Device, DeviceCustom, ModelLoader, Simulator, Synthesizer, Tensor, Random
};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("model.oinf")?;

    let g = graph! {
        dynamic { x: f32[B]; }
        volatile { W(l): f32[D, D] @pattern("W.{l}"); }
        constant { alpha: f32 @ref("alpha"); num_layers: u32; }
        persistent {
            step: i32 @init(0);
            K(l, t): f16[H, Dh] @table;
        }

        block entry {
            assign h: f32[B, D];
            assign cond: bool[];

            op matmul(x, W[0]) >> h;
            op relu(h, negative_slope=0.01, clamp_max=6.0) >> h;
            barrier;

            loop layers (l in 0..num_layers) {
                cache.read K[l, step] >> h;
                op matmul(h, W[l]) >> h;
                cache.write h >> K[l, step];
            }

            op is_finite(h) >> cond;
            branch cond ok bad;
            cache.increment step;
            return;
        }

        block ok {
            op add(h, alpha) >> h;
            yield h;
        }

        block bad {
            op fill(h, value=0.0) >> h;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, Device::Cpu)?
      .with_trace()
      .with_timer();

    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(62846, (-10.0, 10.0), len)?;

    insert_executor!(exec, { x: input });
    exec.run_step()?;

    fetch_executor!(exec, { h: Tensor<f32> });

    let synth = Synthesizer::new(Device::Vulkan);
    let plan = synth.synthesize(&model, &g)?;
    plan.emit("build/out")?;

    Ok(())
}
```
> Some ops in the examples may not yet be implemented; see [docs/progress.md](docs/progress.md) and [docs/ops.md](docs/ops.md) for current support. 

## Philosophy

OpenInfer favors explicit, structured graphs with visible control flow and side effects. See [docs/philosophy.md](docs/philosophy.md) for the full rationale.

## Components

- Model package and OINF format: [docs/oinf-brief.md](docs/oinf-brief.md), [docs/oinf.md](docs/oinf.md)
- DSL and graph construction: [docs/quickstart.md](docs/quickstart.md), [docs/memory.md](docs/memory.md), [docs/control-flow.md](docs/control-flow.md), [docs/cache.md](docs/cache.md)
- Simulation and tracing: [docs/simulation.md](docs/simulation.md)
- Graph serialization: [docs/serialization.md](docs/serialization.md)
- Synthesis and backends: [docs/synthesis.md](docs/synthesis.md), [docs/ops.md](docs/ops.md), [docs/vulkan-interop.md](docs/vulkan-interop.md)
- Implementation notes: [docs/implementation.md](docs/implementation.md)

## Prerequisites

- Rust toolchain (cargo)
- Python 3 + pip (for OINF tools and Python examples)
- Slang compiler (`slangc`) for Vulkan builds (set `SLANGC` or add to PATH)
- Python deps: `pip install -r requirements.txt`

## Build

```bash
cargo build -p openinfer
cargo build -p openinfer --features vulkan
cargo build -p openinfer --features avx,avx2,vulkan
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
python examples/python/{example}_oinf.py
python scripts/verify_oinf.py {example}_model.oinf
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
./scripts/run_rust_examples.sh --target vulkan --features avx,avx2,vulkan
./scripts/run_rust_examples.sh --target all --features avx,avx2,vulkan
```

## Supported Targets

- Architectures: CPU (scalar kernels), CPU SIMD on x86_64, and Vulkan GPU backend
- SIMD extensions: AVX, AVX2 (feature-gated; see [docs/ops.md](docs/ops.md))
- GPU drivers: Vulkan-capable drivers (feature-gated; see [docs/vulkan-interop.md](docs/vulkan-interop.md))
- Vulkan dtype support: `f32`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `bool` (no `f16/f64/bitset`)

## Status

OpenInfer is in early development.

Progress checklist: [docs/progress.md](docs/progress.md)

---


## License

Apache-2.0
