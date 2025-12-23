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

## Core Idea

1. **Models live in a single binary package**

   * Weights and tensors
   * Constants and metadata
   * Shapes, dtypes, layouts
   * Optional persistent buffer definitions

2. **Developers write the execution logic**

   * Inputs and outputs
   * Control flow (loops, branches)
   * Explicit operations
   * Explicit persistent memory access

3. **The result is a symbolic graph**

   * Nothing executes when defined
   * The DSL produces a structured graph of blocks and operations

4. **That graph can be**

   * Simulated for correctness
   * Analyzed and optimized
   * Compiled into device-specific code

---

## Mental Model

Think of OpenInfer as:

> **A small, ML-focused compiler frontend embedded in Rust.**

You describe **what happens and in what order**.
OpenInfer decides **how to execute it efficiently**.

The DSL is closer in spirit to **ONNX / XLA / TVM IRs** than to eager frameworks like PyTorch.

---

## High-Level Workflow

```
Model Package (.oinf)
        │
        ▼
 graph! DSL  ──▶  Symbolic Graph
                        │
                        ├─▶ Simulator (correctness-first)
                        │
                        └─▶ Analyzer / Compiler
                               ▼
                        Device-specific source code
```

---

## Minimal Example

A minimal graph that:

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
            x: f32[B];
        }

        outputs {
            y: f32[B];
        }

        block entry {
            assign t0: f32[B];
            op add(x, x) >> t0;
            op mul(t0, t0) >> y;
            return y;
        }
    };

    let mut sim = SimSession::new(&model, SimBackend::Cpu)?;
    let result = sim.run(&g, inputs! { x: vec![1.0, 2.0, 3.0] })?;

    println!("y = {:?}", result.fetch(out)?);
    Ok(())
}
```

> Variables like `[B]` are named sizes, which are defined in the `model.oinf`; these can be dynamic. The `Simulator` and `Synthesizer` check if the dimensions for the data used with the ops are consistent.

---

## Inputs and Outputs

Inputs and outputs are always declared **at the top level**.

```rust
inputs {
    image: f32[B, 3, H, W];
    step:  i32[];
}

outputs {
    logits: f32[B, C];
}
```

Benefits:

* Shapes and dtypes are explicit
* The graph boundary is well-defined
* Memory and I/O planning is possible up front

---

## Prefix Tables

Many models store repeated tensors under a predictable naming scheme, for example:

* `W.0`, `W.1`, …, `W.9`
* `attn.qkv.0`, `attn.qkv.1`, …

A **prefix table** declares a *family* of model tensors under one DSL name, indexed by one or more symbolic variables.

```rust
prefix {
  W(l): f32[D, D] @pattern("W.{l}");
}
```

How it works:

* `W(l)` declares an indexed handle `W[<expr>]` usable inside blocks.
* `@pattern("W.{l}")` tells the loader how to map an index `l` to a model key.
* Prefix tables are **declarations**: the graph references them, and the runtime resolves them from the model package.

You can also alias different naming schemes:

```rust
prefix {
  QKV(layer): f16[D, 3*D] @pattern("attn.qkv.{layer}");
}
```

---

## Blocks and Execution

Execution logic is written inside **blocks**.

```rust
block entry {
    assign a: f32[B, D];
    op matmul(x, w) >> a;
    op add(a, bias) >> a;
    return a;
}
```

Key properties:

* Blocks execute sequentially
* Each line represents a graph node
* Execution order is explicit
* Blocks end with a terminator (`return`, later `branch`, `yield`, etc.)

Blocks form a **control-flow graph**, not just a flat list of ops.

---

## Assignments and Operations

* `assign` declares a temporary tensor or scalar
* `op` executes a computation and produces an output

```rust
assign h: f32[B, D];
op matmul(x, w) >> h;
```

Assignments are **ephemeral**:

* They exist only during execution
* The runtime may reuse or alias memory
* They do not persist across steps

---

## Loops

Loops are explicit control-flow constructs.

```rust
loop layers (l in 0..num_layers) {
    op matmul(h, W[l]) >> h;
    op relu(h) >> h;
}
```

> Here `layers` is just a name to identify it as a block in the graph; you can use any name like `heads`, `batches`, etc.   

Characteristics:

* Loop bounds are symbolic
* Loop indices are explicit variables
* Loop bodies form nested regions
* Repetition is visible to the compiler

---

## Persistent Memory: Cache

Some inference models require **persistent memory across steps**:

* Transformer KV cache
* Recurrent hidden state
* Streaming buffers
* Rolling windows

OpenInfer models this using **cache**, a generic persistent storage abstraction.

### Declaring a Cache

```rust
cache {
    cursor step: i32 @init(0);
    K(l, t): f16[H, Dh] @table;
    V(l, t): f16[H, Dh] @table;
}
```

Properties:

* Cache lives outside `block entry`
* Cache persists across executions
* Cache entries are indexed
* Cache is architecture-agnostic

---

## Cache Operations

Cache access is explicit and side-effectful.

```rust
cache.read  K[l, step] >> k;
cache.write v >> V[l, step];
cache.advance step;
```

Available primitives:

* `cache.read`
* `cache.write`
* `cache.advance`, `cache.advance {number} x`
* `cache.retract`, `cache.retract {number} x`
* `cache.reset`

This makes data dependencies and ordering explicit and analyzable.

---

## Operation Attributes and Constraints

Attributes let you attach **hints, constraints, or metadata** to:

* Variable definitions (inputs, outputs, constants, prefix tables, cache)
* Control-flow constructs (loops) and blocks

Attributes are written as **function-like annotations**.

### Attributes on variable definitions

Use this for linking model data, expressing layouts, quantization, or other metadata.

```rust
constants {
  alpha: f32 @ref("alpha");
  beta:  f32 @ref("beta");
  bias:  f32 @ref("gamma");
}

prefix {
  W(l): f32[D, D] @pattern("W.{l}") @layout("row_major");
}

cache {
  cursor step: i32 @init(0);
  K(l, t): f16[H, Dh] @table @placement("device");
  V(l, t): f16[H, Dh] @table @placement("device");
}
```

> Some of these attributes like `@layout("row_major");` are the defaults. Also, by default, if the name in the binary file for a variable is the same as in the DSL, you don't need `@ref`. 

### Operator settings are named parameters

Operations do **not** use attributes. Instead, operator configuration is expressed via **named parameters**.

```rust
op relu(h, negative_slope=0.0, clamp_min=0.0, clamp_max=inf) >> h;
```

Examples of realistic activation settings you might see in real deployments:

```rust
// Standard ReLU
op relu(h, negative_slope=0.0) >> h;

// LeakyReLU (common in CNNs)
op relu(h, negative_slope=0.01) >> h;

// Clipped ReLU / ReLU6 (common in mobile / quantization-aware)
op relu(h, negative_slope=0.0, clamp_max=6.0) >> h;

// Lower clamp (occasionally used for numerical stabilization)
op relu(h, negative_slope=0.0, clamp_min=-1e-6) >> h;
```

### What attributes mean

* Attributes are **declarative**: they restrict or guide compiler decisions.
* They do not guarantee a specific implementation.
* Unknown attributes can be preserved (for tooling) or rejected (for strict mode).

---

## Barriers and Control Dependencies

Inference graphs often need explicit ordering boundaries for correctness, debugging, or interoperability.

### Barrier

A `barrier;` prevents motion, fusion, or reordering across the boundary.

```rust
block entry {
  assign h: f32[B, D];

  op matmul(x, W[0]) >> h;
  barrier;
  op relu(h) >> h;
  return h;
}
```

### Explicit control dependency

A control dependency expresses ordering **without creating a data edge**.

This is useful when:

* you must enforce “write happens after compute” even if the value isn’t used later
* you need deterministic traces
* you interface with an external effect that the compiler must not reorder

Example: ensure a cache write happens after an op, but the output tensor itself is not otherwise consumed.

```rust
block entry {
  assign h: f32[B, D];

  op matmul(x, W[0]) >> h;

  // Record the computed activation into a cache table for debugging/inspection.
  // The `dep` makes the ordering explicit even though the write is an effect.
  dep after(matmul) before(cache.write);
  cache.write h >> K[0, step];

  op relu(h, negative_slope=0.0) >> h;
  return h;
}
```

> The syntheziser is free to reorder pure ops, but it must respect explicit deps around effects.

---

## Example: Single Inference Step with Cache

```rust
graph! {
  model: model,

  inputs {
    x: f32[B, D];
  }

  outputs {
    z: f32[B, D];
  }

  prefix {
    W(l): f32[D, D] pattern("W.{l}");
  }

  cache {
    cursor step: i32 @init(0);
    K(l, t): f16[H, Dh] @table;
    V(l, t): f16[H, Dh] @table;
  }

  block entry {
    assign h: f32[B, D];
    assign k: f16[H, Dh];
    assign v: f16[H, Dh];

    transfer x >> h;

    loop layers (l in 0..10) {
      cache.read  K[l, step] >> k;
      cache.read  V[l, step] >> v;

      op attn(h, k, v, W[l]) >> h;

      cache.write k >> K[l, step];
      cache.write v >> V[l, step];
    }

    cache.advance step;

    return h;
  }
}
```

---

## Multiple Steps in the Simulator

Running the graph multiple times advances the cache.

```rust
let mut sim = SimSession::new(&model, SimBackend::Cpu)?;

let mut exec = sim.prepare(&g, inputs! { x: first_token })?;
exec.finish()?;

let mut exec = sim.prepare(&g, inputs! { x: second_token })?;
exec.finish()?;
```

Each invocation:

* Reuses cache contents
* Writes new entries at the next step
* Advances the cache cursor

---

## Simulation

Simulation mode:

* Executes lazily
* Loads model data on demand
* Preserves cache across runs
* Prioritizes correctness and debuggability

Simulation is designed to validate **logic and structure**, not raw performance.

---

## Attributes in Practice

A typical workflow is:

1. Use attributes on **definitions** to connect DSL names to model package names.
2. Use named parameters inside **ops** to select operator behavior.
3. Use barriers / control deps to enforce ordering around effects.

Small example:

```rust
graph! {
  model: model,

  inputs { x: f32[B, D] @layout("row_major"); }
  outputs { z: f32[B, D]; }

  constants {
    scale: f32 @ref("scale");
  }

  prefix {
    W(l): f32[D, D] @pattern("W.{l}");
  }

  block entry {
    assign h: f32[B, D];

    op matmul(x, W[0]) >> h;
    barrier;
    op relu(h, negative_slope=0.0, clamp_max=6.0) >> h;
    op mul(h, scale) >> z;
    return z;
  }
}
```

---

## Branching and Yielding Across Blocks

OpenInfer graphs are **control-flow graphs**. `entry` is always the starting block, but execution can jump to other blocks.

only entry block can assign variables to be used. all subblocks can mutate it, but cannot return value back to entry block. essentially all variables are globals. 

> This is to make parsing and traversing the graph easier. 

### Branch

Use `branch` to jump to another block (optionally based on a condition). 

```rust
block entry {
  assign h: f32[B, D];
  assign cond: bool[];

  op matmul(x, W[0]) >> h;
  op is_finite(h) >> cond;

  branch cond ok bad;
  return h;
}

block ok {
  op relu(h, negative_slope=0.0) >> h;
  return;
}

block bad {
  op fill_nan_like(h, value=0.0) >> h;
  return;
}
```


### Yield

Use `yield {var}` when you want the entry block to remove temporary access to a variable  

This is useful for async or streaming execution.

After yielding, the entry block cannot mutate the variable used by the consuming block. However its free to execute other code. 

Using `await {var}`, multiple blocks can consume the same variable, but only one can mutate it. When multiple block mutate variable, execution will be serialized. 

Entry block has access to the variable whenever all consumers yield the variable. 

```rust
block entry {
  assign h: f32[B, D];
  assign x: i32[D];

  op matmul(x, W[0]) >> h;
  yield x;

  op relu(h, negative_slope=0.0, clamp_max=6.0) >> h;

  await x;
  // do something with x modified by consumer blocks...
  return h;
}

// A different device, core or thread could execute execute this
// The exact scheduling model is backend-defined.

block consumer_1 {
  await x;
  // some compute modifiying x.
  yield x;
}

block consumer_2 {
  await x;
  // some compute reading x.
  yield x;
}

block consumer_3 {
  await x;
  // some compute reading x.
  yield x;
}
```

Notes:

* For sub blocks `yield` is a terminator like `return`. For entry block its an invokation. 
* It defines an explicit control-flow edge to a continuation block.
* Backends may interpret `yield` as “pause and resume”, “send to a queue”, or “schedule on another device”. Implementation depends on device. 

---

## Graph Serialization

Graphs are plain Rust objects and can be serialized to JSON.

```rust
use openinfer::ir::GraphJson;

let json: GraphJson = g.to_json()?;
std::fs::write("graph.json", serde_json::to_string_pretty(&json)?)?;
```

---

## Compilation and Synthesis

Once validated, the same graph can be compiled:

* Device-specific scheduling
* Kernel fusion
* Memory planning
* Backend code generation

The output is **plain source code** (C++, shaders, etc.), not a runtime dependency.

---

## DSL Philosophy

The DSL is intentionally:

* Explicit
* Verbose
* Structured

This ensures:

* Control flow is visible
* Side effects are explicit
* Optimizations are safe
* Generated code is predictable

The DSL describes **intent**, not implementation.

---

## Non-Goals

* Training
* Automatic model conversion
* Implicit execution
* Python-first APIs

---

## Status

OpenInfer is in early development.

Areas open for contribution:

* Graph analysis passes
* Optimization strategies
* Device backends
* Tooling and visualization

---

## License

Apache-2.0



