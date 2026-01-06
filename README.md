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

> **A small, ML-focused Synthesizer frontend embedded in Rust.**

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
                        └─▶ Analyzer / Synthesizer
                               ▼
                        Device-specific source code
```

For more information see [docs/implementation.md](docs/implementation.md)

---

## OINF Binary Format (Brief)

OpenInfer models are stored in a single `.oinf` binary container that holds:

* Named size variables (u64 only)
* Metadata key/value pairs
* Named tensors with dtype, shape, and optional data

Conceptually, an uncompacted view looks like:

```ini
B := 1024
a: f32[B] = { ... }
mode: str = "clamp_up"
```

Create a binary file:

```bash
python examples/python/minimal_oinf.py
```

> See [docs/oinf.md](docs/oinf.md) how you can create your own binary file from a Python Dataclass. 

Inspect it:

```bash
python verify_oinf.py minimal_model.oinf
```

For the full technical spec and layout, see [docs/oinf.md](docs/oinf.md).

---

## Minimal Example

A minimal graph that:

* Takes an input tensor
* Applies two operations
* Returns a result

> Example of `model.oinf` file in non binary format. 
```ini
B := 1024
a: f32[B] = {3.4324, 53.24324, 2334.2345 ...}
```

> minimal.rs
```rust
use openinfer::{
    graph, fetch_executor, insert_executor, Device, ModelLoader, Simulator,
};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("model.oinf")?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            a: f32[B];
            y: f32[B] @init(5.0);
        }

        block entry {
            assign t0: f32[B];
            op add(x, a) >> t0;
            op mul(y, t0) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, Device::Cpu)?;
    let mut exec = sim.make_executor(&g)?;

    let mut rng = rand::thread_rng();
    let len = model.size_of("B")?;
    let input: Vec<f32> = (0..len)
        .map(|i| {
            let base = rng.gen_range(-10.0..=10.0);
            base + (i as f32 * 0.001)
        })
        .collect();

    insert_executor!(exec, { x: input });
    exec.run_step()?;

    fetch_executor!(exec, { y: f32 });
    println!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
```

Variables like `[B]` are named sizes, which are defined in the `model.oinf`; these can be dynamic. The `Simulator` and `Synthesizer` check if the dimensions for the data used with the ops are consistent.

The variables defined in the model binary and the DSL do not need to be exactly the same. The DSL can have new variables not found in the binary, but the DSL cannot have the same variable name with a different data type or dimension. By default the variables are linked between the binary and DSL. So `a: f32[B]` in the DSL is directly linked to `a: f32[B]` in the binary by default. 

### Executor Macros

These macros bridge between user data and the executor. Use the panic-on-error versions for quick scripts, and the `try_*` versions when you want to handle errors yourself.

```rust
insert_executor!(exec, { x: vec![1.0, 2.0, 3.0] });
fetch_executor!(exec, { y: f32 });
println!("y = {:?}", y.data);
```

```rust
try_insert_executor!(exec, { x: vec![1.0, 2.0, 3.0] })?;
let y = try_fetch_executor!(exec, { y: f32 })?;
println!("y = {:?}", y.data);
```

```rust
let (y, z) = try_fetch_executor!(exec, { y: f32, z: i64 })?;
println!("y = {:?}, z = {:?}", y.data, z.data);
```

> The `*_fetch_*` macros require you to specify the type of the tensor you want to load as type information of the graph nodes is loaded during runtime, but the macro runs at compile time. This means you wont know which data type the Tensor is you want to return. You could alternative omit the type hint and use it like `{y}`, but then you need to explicitly turn the TensorWrapper to a Tensor using `y.as_{type}()`. 

---

## Inputs and Outputs

Data used for the model should always defined at the top, before `block entry {}`.

```rust
dynamic {
    x: f32[B];
}

volatile {
    a: f32[B];
    y: f32[B];
}

constant {
    alpha: f32;
}

persistent {
    cache: f16[H, Dh];  
}
```

There are 4 different types of memory that can used with the model:
1. `dynamic`: This is memory that can be mutated by an external program and is cleared every inference step. 
    - This is usefull to feed things like: user input, tokens, weights, images and other data that gets generated on the fly. 
2. `volatile`: This is memory that can be mutated from within a block of logic in the DSL and is filled with the content of the model binary file. Data is reset every inference step. 
    - This is essentially the memory used for things like weights, tensors, etc in the binary file which can be used for performing calculations with the ops. 
3. `constant`: This is similar to `volatile` where data is copied from the model binary, however this memory cannot be mutated, only read. 
    - This is usefull for things like metadata, settings, op params, etc. 
4. `persistent`: This is memory which can be mutated from within a block of logic in the DSL but stays peristant every inference step. This means you can store values from previous inference steps as history.
    - This is usefull for things like KV-cache, rolling windows, recurrent hidden state and anything that needs to persist for every inference step.        


### Interacting with memory
- You can only mutate `dynamic` memory before the inference step is started or has completed. You can use the macro `insert_executor!{}` to modify one or more variables defined in the DSL with new data. 
- You are free to fetch data during any time of the inference step that is stored in { `volatile`, `constant` or `persistent` }, but you **cannot** mutate it. You can use the macro `fetch_executor!{}` to get a copy of one or more variables defined in the DSL. 

---

### Attributes on variable definitions

Atrributes can be used only on variable definitions for things like linking model data, expressing layouts, quantization, or other metadata relevant for the Simulator and Synthesizer.

```rust
constants {
  alpha: f32 @ref("alpha");
  beta:  f32 @ref("beta");
  bias:  f32 @ref("gamma");
}
```

> For example you can use the `@ref` attribute to link a custom variable name in the DSL to a variable name in the binary. 

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

* Attributes are **declarative**: they restrict or guide Simulator and Synthesizer decisions.
* They do not guarantee a specific implementation.
* Unknown attributes can be preserved (for tooling) or rejected (for strict mode).

## Prefix Tables

Many models store repeated tensors under a predictable naming scheme, for example:

* `W.0`, `W.1`, …, `W.9`
* `attn.qkv.0`, `attn.qkv.1`, …

A **prefix table** declares a *family* of model tensors under one DSL name, indexed by one or more symbolic variables.

```rust
volatile {
  W(l): f32[D, D] @pattern("W.{l}");
}
```

How it works:

* `W(l)` declares an indexed handle `W[<expr>]` usable inside blocks.
* The attribute `@pattern("W.{l}")` tells the loader how to map an index `l` to a model key.
* Prefix tables are **declarations**: the graph references them, and the runtime resolves them from the model package.

You can also alias different naming schemes:

```rust
constant {
  QKV(layer, head): f16[D, 3*D] @pattern("attn.{head}.qkv.{layer}");
}
```

Prefix tables can **only** be defined in `volatile` and `constant` memory space. 

---

## Blocks and Execution

Execution logic is written inside **blocks**.

```rust
block entry {
    assign a: f32[B, D];
    op matmul(x, w) >> a;
    op add(a, bias) >> a;
    return;
}
```

Key properties:

* Blocks execute sequentially
* Each line represents a graph node
* Execution order is explicit
* Blocks end with a terminator (`return`, or `yield`)

Blocks form a **control-flow graph**, not just a flat list of ops.

---

## Assignments and Operations

* `assign` declares a temporary tensor or scalar stored in `volatile`. 
* `op` executes a computation and produces an output
* output is alwasy stored in the variable following `>>`. 

```rust
assign h: f32[B, D];
op matmul(x, w) >> h;
```

Assignments are **ephemeral**:

* They exist only during execution
* The runtime may reuse or alias memory
* They do not persist across steps
* The synthizier might optimize them out or use them as reference instead of deep copies. 

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
* Repetition is visible to the Synthesizer

---

## Persistent Memory (a.k.a. "Cache")

Some inference models require **persistent memory across steps**:

* Transformer KV cache
* Recurrent hidden state
* Streaming buffers
* Rolling windows

OpenInfer models this using `persistent` memory, a generic persistent storage abstraction.

### Declaring a Cache

```rust
persistent {
    step: i32 @init(0);
    cache: f16[H, H];
}
```
Properties:

* Cache lives outside `block entry`
* Cache persists across executions
* Cache is architecture-agnostic


### Prefixed cache
Just like a prefix table in `volatile` and `constant` you can create a table layout for `persistent` memory, which you can acces using one or multiple indices. 

This will essentially make a `n` dimensional table for any tensor layout. The table can either be fixed size (for the indices of `n`) or it can dynamically grow. 

This depends on the attributes you set. By default prefix cache will be dynamic.

See examples below.  

```rust
persistent { 
    A(i): f32[D] @table;
    B(i, j): f32[D] @table;
    C(i): f16[D, H] @table;
    D(i, j): f16[D, H] @table @fixed(i=1024, j=256);

    // Example of KV[H, Dh] matrix for each attention head and token. 
    K(l, t): f16[H, Dh] @table;
    V(l, t): f16[H, Dh] @table;
}
```
- `A`: A growable 1D table with `f32[i * D]` elements, accessed like `A[0..i] -> f32[D]`.
- `B`: A growable 2D table with `f32[i * j * D]` elements, accessed like `B[0..i, 0..j] -> f32[D]`.
- `C`: A growable 1D table with `f16[i * D * H]` elements, accessed like `C[0..i] -> f16[D, H]`.
- `D`: A fixed size 2D table table with `f16[1024 * 256 * D * H]` elements, accessed like `D[0..1024-1, 0..256-1] -> f16[D, H]`. 

#### Indicess slice access
You can also access a slice of the table entries. 

For example lets say in a previous step you used `A[10]`, then the prefix cache will contain a table of size `f32[10, D]` (10 columns of D rows). Then in the current step you can access slices of this table like:
- `A[] -> f32[10, D] == A[0..9]`
- `A[0..5] -> f32[5, D]`
- `A[2..5] -> f32[3, D]`
- `A[0..-3] -> f32[7, D] == A[0..6]`

### Autodim cache
In some instances its preferable to have a matrix with an initial fixed size dimension which can grow dynamically in the multiple inference steps. For example in modern LLMs the `Key` and `Value` matrices from previous steps are reused so they dont need to be recomputed. This means the new weights are appended as new columns and rows of the exisiting matrices. 

OpenInfer implement this as a prefix cache using special attribute named `@auto_dim({indices})`. Here you can specificy indicides which are mapped to the dimensions of the tensor. Each inference step a new dimension is allocated for the listed indices in `@auto_dim()`. 

Beware that that access patterns with slices are different than with regular table prefixes, but you can still set the max size of these indices and you can optionally combine autodum with a regular table layout. 

See examples below:
```rust
persistent { 
    A(i, j): f32[D, H] @auto_dim(i, j);
    B(i, j): f32[D, H] @auto_dim(i, j) @fixed(i=1024, j=256);
    C(l, i, j): f32[D, H] @table @auto_dim(i, j);
}
```

- `A`: A growable 2D matrix with `f32[D + i * H + j]` elements, which can be accessed like:
  * `A[0..D+i, ] -> f32[H]`
  * `A[, 0..H+j] -> f32[D]`
  * `A[0..D+i, 0..H+j] -> {i: f32[H+j], j: f32[D+i]}`
  * `A[i, ] -> f32[j]`
  * `A[, j] -> f32[i]`
  * `A[i, j] -> f32[i, j]`
  * `A[] -> f32[D+i, H+j]`

- `B`: Same as `A` but matrix can only be of maximum size `[D+1024, H+256]`.

- `C`: A growable 1D table containing a 2D matrix of size `f32[l * D + i * H + j]`, which has a similar access pattern as `A` but just with an additional index `l` in the beginning like `C[l, i, j]. Essentially you are creating a table of size `l` which contains multiple growable matrices with dimension `f32[D + i, H + j]`. The same sules for Indices slices apply here so using `C[0..4, i, j]` with return a multi-rank tensors with `[4 * i * j]` elements. 

---

## Cache Operations

Cache access is explicit and side-effectful.

```rust
cache.read  K[l, step] >> k;
cache.write v >> V[l, step];
cache.increment step;
cache.increment 5 step;
cache.decrement step;
cache.decrement 2 step;
cache.reset step;
cache.reset K;
cache.reset K[l];
cache.reset K[l, step];
```

Available primitives:

* `cache.read`
* `cache.write`
* `cache.increment`, `cache.increment {number}`
* `cache.decrement`, `cache.decrement {number}`
* `cache.reset`

This makes data dependencies and ordering explicit and analyzable.

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
  return;
}
```

### Explicit control dependency

A control dependency expresses ordering **without creating a data edge**.

This is useful when:

* you must enforce “write happens after compute” even if the value isn’t used later
* you need deterministic traces
* you interface with an external effect that the Synthesizer must not reorder

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
  return;
}
```

> The syntheziser is free to reorder pure ops, but it must respect explicit deps around effects.

---

## Example: Single Inference Step with Cache

```rust
graph! {

  dynamic {
    x: f32[B, D];
  }

  volatile {
    z: f32[B, D];
    W(l): f32[D, D] pattern("W.{l}");
  }

  persistent {
    step: i32 @init(0);
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

    cache.increment step;

    return;
  }
}
```

> `transfer` is not garanteed to be a deep copy, it can be pointer alias, reference or just reusing exisiting variable in `Synthesizer`


### Multiple Steps in the Simulator

Running the graph multiple times advances the cache.

```rust
let mut sim = Simulator::new(&model, Device::Cpu)?;
let exec = sim.make_executor(&g)?;

insert_executor!(exec, { x: first_token });
exec.run_step()?;

insert_executor!(exec, { x: second_token });
exec.run_step()?;
```

Each invocation:

* Reuses cache contents
* Writes new entries at the next step
* Advances the cache cursor

---


## Branching and Yielding Across Blocks

Graphs are **control-flow graphs**. Here `entry` is always the starting block, but execution can jump to other blocks.

Only the entry block can assign variables to be used. All subblocks can mutate it, but cannot return value back to the entry block. Essentially all variables are globals. 

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
  branch algorithm;
  return;
}

block ok {
  op relu(h, negative_slope=0.0) >> h;
  return;
}

block bad {
  op fill_nan_like(h, value=0.0) >> h;
  return;
}

block algorithm {
  // Some sequence of ops changing h...
  return;
}
```

### Yield

Use `yield {var}` when you want the entry block to remove temporary access to a variable  

This is useful for async or streaming execution.

After yielding, the entry block cannot mutate the variable used by the consuming blocks. However its free to execute other code. 

Using `await {var}`, multiple blocks can consume the same variable, but only one can mutate it.  

Entry block has access to the variable whenever all consumers yield the variable. 

```rust
block entry {
  assign h: f32[B, D];
  assign x: i32[D];

  op matmul(x, W[0]) >> h;
  yield x; //x not available anymore to entry

  // These ops are executed in parrallel
  op relu(h, negative_slope=0.0, clamp_max=6.0) >> h;

  // Waitng for all consumers to be done.
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

* For sub blocks `yield` is a terminator like `return`. For the entry block its an invokation. 
* It defines an explicit control-flow edge to a continuation block.
* Backends may interpret `yield` as “pause and resume”, “send to a queue”, or “schedule on another device”. Implementation depends on device. 
* Each consumer will have the last know value of the variable that was yielded by the entry block. This means that the consumer which mutates x will not affect other consumers which read x. 

> You can `yield` and `await` multiple variable like: `yield a, b, c;` and `await a, b, c;`. In this case the `await` will be serialized untill all variables are available. This also means multiple blocks can mutate different variables, but the rule of 1 block per variable still applies. So for example `b1` mutates `a` and `b2` mutates `b, c`.   

---

## Simulation

Simulation mode:

* Executes lazily
* Loads model data on demand
* Preserves cache across runs
* Prioritizes correctness and debuggability

Simulation is designed to validate **logic and structure**, not raw performance.

### Step through nodes with logging + timing

```rust
use openinfer::{
  fetch_executor, format_truncated, graph, insert_executor, 
  Device, ModelLoader, Simulator
};

...

let sim = Simulator::new(&model, Device::Cpu)?;
let mut exec = sim.make_executor(&g)?;
insert_executor!(exec, { x: input });

// This is equivalent to what happens when you call exec.step().
for mut node in exec.iterate() {
    let ev = node.event.clone();
    fetch_executor!(node, { y: f32 });
    let y_str = format_truncated(&y.data);
    let y_pad = format!("{:<width$}", y_str, width = 32);
    println!(
        "y={} -- [{}] {} :: {} ({})",
        y_pad,
        ev.kind,
        ev.block_name,
        ev.op_name,
        ev.micros
    );
}
```

### Export a simulator trace

```rust
let trace = exec.trace();
std::fs::write("build/trace.json", serde_json::to_string_pretty(&trace)?)?;
```

Example of a trace output:
```json
[
  {
    "block_name": "entry",
    "node_index": 0,
    "node_uuid": "f4137d97-0919-4f73-9839-dd74367f943c",
    "kind": "Assign",
    "params": [],
    "output": ["t0"],
    "micros": [0, 0, 0]
  },
  {
    "block_name": "entry",
    "node_index": 1,
    "node_uuid": "53b7184e-4b0f-492d-8d93-9274d52617a6",
    "kind": "OpExecute",
    "params": ["x", "a"],
    "output": ["t0"],
    "micros": [0, 31, 788]
  },
  {
    "block_name": "entry",
    "node_index": 2,
    "node_uuid": "16ce2716-d6fc-4136-bd6f-1b8d5a66669e",
    "kind": "OpExecute",
    "params": ["y", "t0"],
    "output": ["y"],
    "micros": [0, 30, 12]
  },
  {
    "block_name": "entry",
    "node_index": 3,
    "node_uuid": "8500432b-f1c3-4d14-86de-af5c8f97506b",
    "kind": "Return",
    "params": [],
    "output": [],
    "micros": [0, 0, 0]
  }
]
```

---

## Graph (De)Serialization

Graphs are plain Rust objects and can be serialized to JSON.

Serialization (see `examples/rust/serialize.rs`):
```rust
let json = GraphSerialize::json(&g)?;
std::fs::write("minimal-graph.json", serde_json::to_string_pretty(&json)?)?;
```

Deserialization (see `examples/rust/deserialize.rs`):
```rust
let value = serde_json::from_str(&graph_txt)?;
let g = GraphDeserialize::from_json(value)?;
```

---

## Backend Documentation

See backend-specific docs for implementation details and extension points:

- `docs/vulkan-interop.md`

---

## Supported Ops

For a current list of supported ops and per-backend dtype coverage, see:

- `docs/ops.md`

---

## Synthesis

Once validated, the same graph can be synthezised to source code with:

* Device-specific scheduling
* Kernel fusion
* Memory planning
* Backend code generation

The output is **plain source code** (C, GLSL shaders, etc.), **not** a binary.

### Passing Graphs to the Synthesizer

The synthesizer accepts either:

* a `Graph` object directly (in-process), or
* a serialized JSON graph (tooling / CLI / build pipelines)

#### In-process usage

```rust
use openinfer::{Synthesizer, Device};

let dev = Device::Vulkan;
let synth = Synthesizer::new(dev);

let plan = synth.synthesize(&model, &graph)?;
plan.emit("build/out")?;
```

### Device Architecture JSON (Synthesizer Input)

For reproducible compilation, the synthesizer can be configured with a JSON file that describes the target device architecture.

This file is intended to be:

* explicit (no guessing)
* stable in CI/build systems
* extensible over time

> Mock example: `ada_mock.json`

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

#### Loading the device JSON

```rust
use openinfer::{Synthesizer, DeviceCustom};

let txt = std::fs::read_to_string("devices/ada_mock.json")?;
let arch: DeviceCustom = serde_json::from_str(&txt)?;

let synth = Synthesizer::from_arch(arch);
let plan = synth.synthesize(&model, &g)?;
plan.emit("build/out")?;
```

> The device JSON is how you make compilation **repeatable** across machines and CI environments.


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

Parsing of the DSL has begun, final touches on changing the DSL specs will be made before parsing to graph layout is finalized. 
Implementations for Ops on CPU and GPU are already developed in C, just need porting to Rust and integration with this codebase. 

---

## License

Apache-2.0
