```
 ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░       ░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░ ░▒▓███████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
 ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░ 
                                                                                                                       
                                                                                                                       

```

**OpenInfer** is an open-source inference library for building **lazy, step-by-step inference graphs** on top of a **single binary model package** (GGUF-like), and later **compiling those graphs into optimized executables** for specific hardware targets.

OpenInfer is designed to make **developer ergonomics the priority**, while allowing the implementation to grow into a full **compiler-style optimization pipeline** underneath.


---

## Core Idea

1. **All tensors, metadata, and state specifications live in a single binary file**

   * Weights, constants, quantization info
   * Model metadata (dimensions, hyperparameters)
   * KV-cache specifications (layout, capacity)
2. **Developers define only the execution graph**

   * Which ops to run
   * Which tensors/metadata from the model package to use
3. The graph is:

   * **Lazy** in simulation mode (correctness-first)
   * **Optimized and compiled** later for a specific backend (CPU, Vulkan GPU, future FPGA)

This workflow is inspired by GGML/GGUF-style inference, but with a **graph + compiler architecture** instead of a fixed runtime.

However, OpenInfer differs from existing inference libraries like llama-cpp, as the goal is not to predefine all existing models with a generic interface to deploy their models, but to allow you to create and define your own inference model using a programmatic approach.

So a realistic workflow would be: 
1. You train your own model using PyTorch or a similar library for ML. 
2. You export the model to some format like `HuggingFace` with safetensors and metadata.
3. You use a conversion script to compress all needed data to a binary format (similar to `GGUF`) named `OPNF` (Open Infer Neural Format)
4. You design the inference loop generally using Rust to create a `Graph` object.
5. You pass this object to a simulator, which executes the nodes lazily, loading tensors from disk to memory when necessary.
    - In the future GPU backend (`Vulkan` only) will also be supported to speed up simulation OPS with large weights.
    - The simulator is used for correctness, not speed. You are just verifying that the inference loop you designed works correctly with the data.
    - The reason for lazy evaluation is that on memory-constrained systems, you are not able to load all of the tensors and weights into memory; you need to reuse the same buffer, but with a different chunk of the matrices.  
7. After you are satisfied with the results of the simulator, you pass this object to the `Evaluator` to generate an optimized solution for your specified hardware.
   - This will be a simple source file that you can feed to the toolchain used for the hardware to compile into the desired binary executable format.
   - Optimizations are on a device-by-device level (not only device type, ex, CPU, CPU, NPU, etc). Initially, the number of supported devices will be limited. 
   - For example, if you export to two different types of GPU architectures or vendors, the output file might be different to take advantage of Vulkan vendor-specific extensions.

---

## Goals

* Concise, readable graph definitions
* Lazy simulation for correctness and debugging
* Compiler-style optimization passes
* Backend abstraction (CPU, Vulkan compute, future accelerators)
* Everything references a **single binary model package**

---

## Non-Goals (for now)

* Training
* Automatic model conversion
* Python-first APIs (bindings can come later)

---

## Model Package

OpenInfer expects a single binary model file (e.g. `model.oinf`) containing:

* **Tensor table**

  * Key → dtype, shape, quantization, file offset
* **Metadata store**

  * Typed values (e.g. `n_layers`, `ctx_len`, `rope_theta`)
* **State specs**

  * KV-cache layout and capacity (contents allocated at runtime)
* *(Optional, future)* Embedded graph IRs

> **Important:** Tensor data is loaded lazily. Nothing is read from disk until it is actually needed during execution. Buffers are shared between passes if memmory is limited.

---

## High-Level Workflow

```
ModelLoader ──▶ Graph (lazy, symbolic)
                   │
                   ├─▶ SimSession (CPU / Vulkan)
                   │      correctness-first
                   │
                   └─▶ Optimizer / Compiler
                          ↓
                    ExecutablePlan (CPU / Vulkan / FPGA)
```

---

## Minimal Example

A very small inference graph that:

* loads embeddings and lm_head from the model package
* runs a single matmul
* returns logits

```rust
use openinfer::{ModelLoader, SimSession, SimBackend, graph};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("model.oinf")?;

    let (g, logits) = graph! {
        model: model,

        inputs: {
            tok: i32[B],
        },

        let x      = gather(model["tok_embeddings"], tok) |> cast(f16);
        let logits = matmul(x, model["lm_head"]);

        output logits;
    };

    let mut sim = SimSession::new(&model, SimBackend::Cpu)?;
    let out = sim.run_step(&g, inputs! { tok: vec![42] })?;

    println!("logits = {:?}", out.fetch(logits)?);
    Ok(())
}
```

---

## KV Cache + Loop Example

A concise transformer-style decode step:

* reads metadata from the model file
* declares a KV cache
* loops over layers
* performs attention and projection

```rust
use openinfer::{
    ModelLoader, SimSession, SimBackend,
    Evaluator, DeviceSpec, HostLang, graph,
};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("model.oinf")?;

    let n_layers: usize = model.meta("n_layers")?;
    let ctx_len: usize  = model.meta("ctx_len")?;

    let (g, logits) = graph! {
        model: model,

        inputs: {
            tok:  i32[B],
            step: i32[],
            mask: u8[B,1,1,ctx_len],
        },

        state: {
            kv: kv_cache(f16, layers=n_layers, cap=ctx_len),
        },

        let x = gather(model["tok_embeddings"], tok) |> cast(f16);

        let x = for l in 0..n_layers {
            let q = matmul(x, model[format!("Wq.{l}")]);
            let k = matmul(x, model[format!("Wk.{l}")]);
            let v = matmul(x, model[format!("Wv.{l}")]);

            kv_write(kv, layer=l, step=step, k=k, v=v);
            let (k_all, v_all) = kv_read(kv, layer=l, step=step);

            let a = sdpa(q, k_all, v_all, mask) @ { fuse: true };
            let y = matmul(a, model[format!("Wo.{l}")]);

            x = x + y;
        };

        let logits = matmul(x, model["lm_head"]);
        output logits;
    };

    // Lazy simulation (correctness-first)
    let mut sim = SimSession::new(&model, SimBackend::Cpu)?;
    let _ = sim.run_step(&g, inputs! { tok: vec![42], step: 0, mask: /* ... */ })?;

    // Device-specific evaluation (Vulkan-only, GPU-specialized)
    let dev = DeviceSpec::GPU()
        .api("vulkan")
        .vendor("NVIDIA")
        .architecture("ada_lovelace")
        .driver("555.xx");

    let eval = Evaluator::new(dev).from_graph(&model, &g)?;

    // Emits one or more device shader sources (GLSL/HLSL/etc. chosen by Evaluator)
    // Evaluator decides how many shaders to generate and their filenames.
    eval.emit_device("build/device")?;

    // Emits host-side source that:
    // - initializes Vulkan
    // - creates pipelines/descriptors
    // - uploads weights from the model package
    // - manages KV buffers and dispatch order
    // - exposes a simple `run_step()` interface
    //
    // User can choose the host language/tooling.
    eval.emit_host(HostLang::Cpp, "build/host")?;

    Ok(())
}

```

---

## Simulation Mode

* Graph is evaluated lazily
* Only required nodes are executed
* Weights are loaded from disk on first use and cached
* KV cache persists across steps
* Minimal optimization (easy debugging)

---

## Compilation / Optimization Mode

* Graph analysis and rewrites (fusion, scheduling, layout)
* Backend-specific lowering

  * CPU kernels
  * Vulkan compute pipelines
* Produces a reusable `ExecutablePlan`

---

## Macro DSL Philosophy

The macro DSL is **only a frontend**:

* Expands into normal Rust API calls
* Builds a symbolic graph (no execution)
* Keeps user code concise and readable

Features:

* `model["key"]` → tensor reference by key
* `|>` pipe for chaining ops
* `@ { ... }` for metadata and optimization hints
* Structured control flow (`for`, future `if`)

---

## Backend Support

* CPU (reference + optimized)
* Vulkan compute (GPU backend)
* Future: FPGA / custom accelerators

---

## Project Status

Early-stage design / prototype.

Contributions welcome:

* Graph IR and passes
* CPU reference kernels
* Vulkan compute kernels
* Model package format and tooling

---

## License

Apache-2.0

