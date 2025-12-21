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

OpenInfer is an open‑source **inference graph and compilation framework** for machine‑learning workloads.

Its primary goal is to let **developers describe inference logic and control flow explicitly**, using a clear, verbose, Rust‑embedded DSL, while OpenInfer handles simulation, analysis, optimization, and code generation for specific hardware targets.

OpenInfer is **not model‑specific**. The same DSL can describe inference for:

* LLMs and transformers
* CNNs and vision models
* Audio / signal‑processing models
* Custom or experimental ML architectures

The focus is on **developer control, readability, and inspectability**, not on hiding complexity behind opaque runtimes.

---

## Core Idea

1. **Models live in a single binary package**

   * Weights, constants, metadata
   * Shapes, dtypes, quantization
   * Optional persistent state layouts

2. **Developers write the execution logic**

   * Inputs and outputs
   * Control flow (loops, branches)
   * Explicit operations and side effects
   * Constraints and optimization hints

3. **The graph is symbolic**

   * Nothing executes when defined
   * The result is a structured graph of blocks and operations

4. **That graph can be**

   * Simulated lazily for correctness
   * Analyzed and optimized
   * Compiled into device‑specific executables

---

## Mental Model

Think of OpenInfer as:

> *A small, ML‑focused compiler frontend embedded in Rust.*

You describe **what happens** and **in what order**.
OpenInfer decides **how to execute it efficiently** on the chosen hardware.

---

## High‑Level Workflow

```
Model Package (.oinf)
        │
        ▼
 graph! DSL  ──▶  Symbolic Graph (blocks + ops)
                        │
                        ├─▶ Simulator (lazy, correctness‑first)
                        │
                        └─▶ Evaluator / Compiler
                               ▼
                        Device‑specific source code
```

---

## Minimal Example

A very small graph that:

* Takes an input tensor
* Applies two operations
* Returns a result

```rust
use openinfer::{ModelLoader, SimSession, SimBackend, graph};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("model.oinf")?;

    let (g, out) = graph! {
        model: model,

        inputs {
            x : f32 [B];
        }

        outputs {
            y : f32 [B];
        }

        block entry {
            t0 = op add(a=x, b=x);
            y  = op mul(a=t0, b=model["scale"]);
            return y;
        }
    };

    let mut sim = SimSession::new(&model, SimBackend::Cpu)?;
    let result = sim.run(&g, inputs! { x: vec![1.0, 2.0, 3.0] })?;

    println!("y = {:?}", result.fetch(out)?);
    Ok(())
}
```

---

## Inputs and Outputs

Inputs and outputs are always declared **at the top level**.

```rust
inputs {
    image : f32 [B, 3, H, W];
    step  : i32 [];
}

outputs {
    logits : f32 [B, C];
}
```

Benefits:

* Shapes and dtypes are explicit
* The graph boundary is well‑defined
* The evaluator can plan memory and I/O early

---

## Blocks and Control Flow

Execution logic is written in **blocks**.

```rust
block entry {
    a = op matmul(a=x, b=w);
    b = op add(a=a, b=bias);
    return b;
}
```

Blocks:

* Execute sequentially
* Contain operations and effects
* End with an explicit terminator (`return`, `branch`, `yield`)

Blocks form a **control‑flow graph**, not just a flat list of ops.

---

## Loops

Loops are explicit control‑flow constructs.

```rust
loop layers (l in 0..num_layers) {
    block body {
        h = op matmul(a=h, b=model[fmt("W.{l}")]);
        h = op relu(x=h);
    }
}
```

Characteristics:

* Loop bounds are symbolic
* Loop bodies are normal blocks
* Loop‑carried values must be explicit

This makes iteration visible and analyzable by the compiler.

---

## Side Effects and State

Operations that mutate or read persistent state are written as **effects**.

```rust
state {
    cache : buffer(f32, [B, T, D]);
}

block entry {
    effect write(cache, index=step, value=h);
    h2 = effect read(cache, index=step - 1);
    return h2;
}
```

Why effects are explicit:

* Prevents illegal reordering
* Makes dependencies visible
* Enables correct scheduling and fusion

---

## Operation Attributes and Constraints

Operations and blocks can carry **constraints and hints**.

```rust
a = op matmul(a=x, b=w)
      with { fuse = true, placement = "gpu" };
```

Common constraints:

* `fuse = true | false`
* `placement = "cpu" | "gpu"`
* `group = "name"`
* `no_reorder = true`

Constraints:

* Do not guarantee a specific implementation
* Restrict or guide optimizer decisions
* Are preserved through compilation

---

## Barriers and Control Dependencies

You can prevent motion or fusion across boundaries using barriers.

```rust
block entry {
    a = op matmul(a=x, b=w);
    barrier;
    b = op add(a=a, b=bias);
    return b;
}
```

Use cases:

* Debugging
* Interfacing with external state
* Enforcing ordering for numerical or memory reasons

---

## Scopes and Regions

Scopes annotate logical regions of the graph.

```rust
scope "attention" {
    a = op sdpa(q=q, k=k, v=v);
    b = op matmul(a=a, b=wo);
}
```

Scopes:

* Group related operations
* Aid diagnostics and visualization
* Provide natural fusion or optimization boundaries

---

## Simulation

Simulation mode:

* Executes lazily
* Loads weights on demand
* Preserves state across steps
* Prioritizes correctness and debuggability

Simulation is meant to validate **logic**, not performance.

---

## Graph Serialization (JSON)

Graphs are regular Rust objects and can be **serialized to JSON** for:

* reproducible runs
* CI snapshots / golden tests
* offline analysis and visualization
* passing graphs between tools (simulator ⇄ synthesizer)

### Serialize a graph

```rust
use openinfer::{ModelLoader, graph, ir::GraphJson};

let model = ModelLoader::open("model.oinf")?;
let (g, y) = graph! {
    model: model,

    inputs { x: f32 [B]; }
    outputs { y: f32 [B]; }

    block entry {
        t0 = op add(a=x, b=x);
        y  = op mul(a=t0, b=model["scale"]);
        return y;
    }
};

// Convert to a stable JSON representation and write to disk
let json: GraphJson = g.to_json()?;
std::fs::write("build/graph.json", serde_json::to_string_pretty(&json)?)?;
```

### Load a graph from JSON

```rust
use openinfer::{ir::GraphJson, Graph};

let txt = std::fs::read_to_string("build/graph.json")?;
let json: GraphJson = serde_json::from_str(&txt)?;

// Reconstruct the graph object
let g: Graph = Graph::from_json(json)?;
```

> Tip: JSON is intended to be **stable and inspectable**. It should preserve block structure, op attributes, and control-flow edges.

---

## Simulator: Step-by-step Execution (Verbose)

In addition to running the full graph, the simulator can:

* step through execution **block-by-block** or **node-by-node**
* emit **verbose logs** (op name, shapes, attrs)
* record **timings** for each node (useful for debugging and early performance signals)

### Run a full graph

```rust
use openinfer::{SimSession, SimBackend};

let mut sim = SimSession::new(&model, SimBackend::Cpu)?;
let out = sim.run(&g, inputs! { x: vec![1.0, 2.0, 3.0] })?;
println!("y = {:?}", out.fetch(y)?);
```

### Step through nodes with logging + timing

```rust
use openinfer::{SimSession, SimBackend, sim::TraceLevel};

let mut sim = SimSession::new(&model, SimBackend::Cpu)?
    .with_trace(TraceLevel::Verbose)
    .with_timing(true);

let mut exec = sim.prepare(&g, inputs! { x: vec![1.0, 2.0, 3.0] })?;

while exec.is_running() {
    let ev = exec.step()?;

    // Typical step event info
    // - block/op identifiers
    // - input/output shapes
    // - duration (if enabled)
    println!(
        "[{}] {} :: {}  ({} µs)",
        ev.kind,          // BlockEnter | OpExecute | BlockExit | ...
        ev.block_name,    // "entry" etc.
        ev.op_name,       // "matmul" etc. (empty for non-op events)
        ev.micros
    );
}

let out = exec.finish()?;
println!("final y = {:?}", out.fetch(y)?);
```

### Export a simulator trace

```rust
let trace = exec.trace();
std::fs::write("build/trace.json", serde_json::to_string_pretty(&trace)?)?;
```

---

## Passing Graphs to the Synthesizer

The synthesizer accepts either:

* a `Graph` object directly (in-process), or
* a serialized JSON graph (tooling / CLI / build pipelines)

### In-process usage

```rust
use openinfer::{Synthesizer, DeviceSpec};

let dev = DeviceSpec::GPU().api("vulkan");
let synth = Synthesizer::new(dev);

let plan = synth.synthesize(&model, &g)?;
plan.emit("build/out")?;
```

### From JSON

```rust
use openinfer::{Synthesizer, ir::GraphJson, Graph};

let dev = DeviceSpec::GPU().api("vulkan");
let synth = Synthesizer::new(dev);

let graph_txt = std::fs::read_to_string("build/graph.json")?;
let graph_json: GraphJson = serde_json::from_str(&graph_txt)?;
let g = Graph::from_json(graph_json)?;

let plan = synth.synthesize(&model, &g)?;
plan.emit("build/out")?;
```

---

## Device Architecture JSON (Synthesizer Input)

For reproducible compilation, the synthesizer can be configured with a JSON file that describes the target device architecture.

This file is intended to be:

* explicit (no guessing)
* stable in CI/build systems
* extensible over time

### Mock example

```json
{
  "device": {
    "type": "gpu",
    "api": "vulkan",
    "vendor": "nvidia",
    "name": "Mock RTX",
    "architecture": "ada_lovelace",
    "driver": "555.xx"
  },
  "limits": {
    "max_workgroup_size": [1024, 1024, 64],
    "max_shared_memory_bytes": 65536,
    "max_push_constants_bytes": 256,
    "max_storage_buffer_range_bytes": 2147483647
  },
  "features": {
    "fp16": true,
    "int8": true,
    "subgroup_ops": true,
    "cooperative_matrix": false
  },
  "memory": {
    "global_bytes": 17179869184,
    "shared_bytes_per_sm": 65536,
    "l2_bytes": 67108864
  },
  "preferences": {
    "default_precision": "fp16",
    "prefer_fusion": true,
    "prefer_persistent_kv": true,
    "max_kernel_ops": 12
  }
}
```

### Loading the device JSON

```rust
use openinfer::{Synthesizer, device::DeviceArch};

let txt = std::fs::read_to_string("devices/ada_mock.json")?;
let arch: DeviceArch = serde_json::from_str(&txt)?;

let synth = Synthesizer::from_arch(arch);
let plan = synth.synthesize(&model, &g)?;
plan.emit("build/out")?;
```

> The device JSON is how you make compilation **repeatable** across machines and CI environments.

---

## Evaluation and Compilation

After validation, the same graph can be compiled:

* Device‑specific scheduling
* Kernel fusion
* Memory planning
* Backend‑specific code generation

The output is **plain source code** (C++, shaders, etc.), not a runtime dependency.

---

## DSL Philosophy

The DSL is intentionally:

* Verbose
* Explicit
* Structured

This makes:

* Control flow visible
* Dependencies analyzable
* Optimizations safe
* Generated code predictable

The DSL describes **intent**, not implementation.

---

## Non‑Goals

* Training
* Automatic model conversion
* Implicit execution or magic scheduling
* Python‑first APIs

---

## Status

OpenInfer is in early development.

Areas open for contribution:

* Graph analysis passes
* Optimization strategies
* Device backends
* Tooling and visualization

The DSL and IR are expected to evolve, but the core philosophy—**developer‑authored inference logic with compiler‑level optimization**—is stable.



## License

Apache-2.0

