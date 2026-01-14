# Implementation Guide

This document explains how the DSL expands into a `Graph`, how `.oinf` files are
loaded and consumed lazily, how model variables are linked to DSL declarations,
and how ops are selected at runtime. The goal is to help new contributors follow
the execution path from `graph!{}` to kernel selection.

## Simulator Tracing and Timing

The simulator does not emit trace logs or time ops by default. Enable tracing
and timing explicitly on construction, for example
`Simulator::new(&model, &graph, Device::Cpu)?.with_trace().with_timer()`. Trace events
are only stored when tracing is enabled. The simulator retains the validated
graph, so `make_executor()` no longer needs it as an argument.
During construction the simulator validates the graph against the model (sizevars,
dtype compatibility, constant mutation, scalar-only attributes) and stores the
validated graph for executor creation.

## DSL Parsing -> Graph

The DSL lives in `openinfer-dsl/src/lib.rs` and is implemented as a procedural
macro `graph! { ... }`. The macro uses `syn` to parse a small grammar and then
expands it into Rust code that constructs a runtime `Graph`.

### DSL structure

With the current level of support of the DSL it has two top-level sections:

- Memory sections: `dynamic { ... }`, `volatile { ... }`, `constant { ... }`, or `persistent { ... }`
- Blocks: `block entry { ... }` with nodes inside

Node types inside a block:

- `assign name: dtype[Dims];` declares a temporary variable
- `op add(x, y) >> out;` invokes an op
- `cache.read`, `cache.write`, `cache.increment`, `cache.decrement`, `cache.reset` for persistent cache access
- `loop name (i in start..end) { ... }` for repeated blocks
- `return;` stops the block

> Please not as the new features of the DSL are being added this guide will change.

### Parsing and expansion

Parsing happens in `impl Parse for GraphDsl` and the helpers in
`openinfer-dsl/src/lib.rs`. Each memory section parses variable declarations
(`VarDecl`) that include `dtype`, optional `dims`, and optional `@init(...)`.
Persistent variables may additionally carry `@table`, `@auto_dim(...)`, and
`@fixed(...)` attributes which are stored on the declaration.
Each block parses nodes into one of `Assign`, `Op`, cache operations, `Loop`, or `Return`.

Expansion is in `GraphDsl::expand`:

- Creates a new `Graph` via `Graph::new()`
- Adds variables with `Graph::add_var(...)` including prefix-table metadata
  (`pattern`, `table_indices`) and cache metadata (`table`, `auto_dim`, `fixed`)
- Adds blocks with `Graph::add_block(name)`
- Adds nodes with `Graph::add_node(block, NodeKind::...)`

The DSL is thus a thin front-end that generates ordinary Rust code which builds
the `Graph` value at runtime.

### What the Graph looks like

`Graph` is defined in `openinfer/src/graph.rs` and is serializable via serde.
It has:

- `vars: HashMap<String, VarDecl>` describing memory, dtype, dims, optional init
  values, and cache attributes (`table`, `auto_dim`, `fixed`) when applicable.
- `blocks: HashMap<String, Block>` with ordered nodes.
- `next_index` used to assign monotonically increasing node indices.

Each `Block` holds an ordered list of `Node`, and each `Node` contains:

- `index`: the order in which it was inserted
- `uuid`: a unique identifier
- `kind`: one of `Assign`, `Op`, `CacheRead`, `CacheWrite`, `CacheIncrement`,
  `CacheDecrement`, `CacheReset`, `Loop`, or `Return`

Example (conceptual):

```rust
Graph {
  vars: {
    "x" => VarDecl { name: "x", dtype: F32, dims: ["B"], kind: Dynamic, init: None },
    "a" => VarDecl { name: "a", dtype: F32, dims: ["B"], kind: Volatile, init: None },
  },
  blocks: {
    "entry" => Block {
      name: "entry",
      nodes: [
        Node { index: 0, uuid: "...", kind: Assign { name: "t0", dtype: F32, dims: ["B"] } },
        Node { index: 1, uuid: "...", kind: Op { op: Add, attrs: None, inputs: ["x","a"], output: "t0" } },
        Node { index: 2, uuid: "...", kind: Return },
      ],
    },
  },
}
```

## Loading `.oinf` and Lazy Access

The `.oinf` loader is `ModelLoader` in `openinfer/src/model_loader.rs`.
`ModelLoader::open(path)` does a full header + index parse, but does not load
tensor payloads into memory. Instead, it stores offsets and sizes for each
tensor, so payloads are fetched only when needed.

### What is parsed

`ModelLoader::open` reads the file into memory once and validates:

- Magic/version and header integrity
- Ascending, aligned section offsets
- Sizevars table
- Metadata index table
- Tensor index table

For each tensor entry in the tensor index, it stores:

- Name
- DType
- Dims (as strings)
- `value_range` byte offsets if `HAS_DATA` is set
- `has_data` flag

This becomes `ModelLoader.vars: HashMap<String, VarInfo>`.

### Lazy loading path

Lazy access happens in the `Executor` (`openinfer/src/simulator/executor.rs`):

1. `Executor::new` sets non-dynamic variables to `StoredTensor::Unloaded`.
2. When an op needs an input, `Executor::get_tensor` is called.
3. If the tensor is `Unloaded`:
   - If the name exists in the `.oinf` index and `has_data`, load with
     `ModelLoader::load_tensor` which seeks into the file and reads only that
     byte range.
   - If there is no data blob, fall back to DSL init (`VarDecl::init`) or
     allocate zeros with the backend allocator.
4. The loaded/allocated data is stored as `StoredTensor::Data` and reused.

This makes model parameters lazy: nothing is read from disk until a node
actually consumes the variable.

## Linking `.oinf` Variables to `graph!{}` Variables

The link is by **name** and **dims**:

- The DSL declares variables (`Graph.vars`) by name.
- The `.oinf` file declares tensors (`ModelLoader.vars`) by name.
- At runtime, `Executor::get_tensor(name)` checks `ModelLoader::var_info(name)`
  to decide whether that variable has on-disk data and how big it is.

Dimensions in the graph are stored as strings. They can be:

- Numeric literals: `f32[128]` -> `"128"`
- Sizevar names: `f32[B]` -> `"B"`

`ModelLoader::resolve_len(dims)` resolves these at runtime:

- For numeric strings, parse directly.
- For named dims, look up the sizevar in the `.oinf` header.

If a DSL variable references a sizevar missing from the `.oinf` file, or if a
tensor name does not exist in the `.oinf` index, the executor returns an error.
This keeps the linkage explicit and deterministic.

## Op Selection from Graph Nodes

Runtime op selection flows through `Executor::exec_op`:

1. The `NodeKind::Op` from the graph contains:
   - `op` (e.g. `Add`, `Mul`, `Abs`)
   - `attrs`
   - `inputs` (names)
   - `output`
2. `exec_op` loads inputs via `get_tensor`, then infers `input_dtypes`.
3. `output_dtype` is taken from the graph’s var declaration if it exists;
   otherwise it falls back to the first input dtype.
4. The backend chooses a kernel via `ops::lookup_kernel` based on:
   - `Device` (Cpu / CpuAvx / CpuAvx2 / Vulkan)
   - `OpKind` and `OpAttrs`
   - `output_dtype` and `input_dtypes`

This dispatch is centralized in `openinfer/src/ops/registry.rs`, which forwards
to device-specific registries (e.g. `openinfer/src/ops/cpu/registry.rs`).

### Example serialized graph

> `examples/rust/out/minimal-graph.json`
```json
{
  "blocks": {
    "entry": {
      "name": "entry",
      "nodes": [
        {
          "index": 0,
          "kind": {
            "Assign": {
              "dims": [
                "B"
              ],
              "dtype": "F32",
              "name": "t0"
            }
          },
          "uuid": "06133eae-65c0-4e27-a7c3-4a6add5406b8"
        },
        {
          "index": 1,
          "kind": {
            "Op": {
              "attrs": "None",
              "inputs": [
                "x",
                "a"
              ],
              "op": "Add",
              "output": "t0"
            }
          },
          "uuid": "8895eff0-b5a7-42c4-8acd-b3c6330ddc19"
        },
        {
          "index": 2,
          "kind": {
            "Op": {
              "attrs": "None",
              "inputs": [
                "y",
                "t0"
              ],
              "op": "Mul",
              "output": "y"
            }
          },
          "uuid": "bc609943-4c7c-4782-8a49-9248a37da8bc"
        },
        {
          "index": 3,
          "kind": "Return",
          "uuid": "32131024-3b2d-4877-a27c-25369e7840d6"
        }
      ]
    }
  },
  "next_index": 4,
  "vars": {
    "a": {
      "dims": [
        "B"
      ],
      "dtype": "F32",
      "init": null,
      "kind": "Volatile",
      "name": "a"
    },
    "x": {
      "dims": [
        "B"
      ],
      "dtype": "F32",
      "init": null,
      "kind": "Dynamic",
      "name": "x"
    },
    "y": {
      "dims": [
        "B"
      ],
      "dtype": "F32",
      "init": {
        "F32": 5.0
      },
      "kind": "Volatile",
      "name": "y"
    }
  }
}
```

This node carries everything needed to select a kernel: the op type, attributes,
input/output names (which map to dtypes and memory kinds), and the current
device.
