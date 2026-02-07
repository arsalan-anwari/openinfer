Core Concepts
=============

This page consolidates the simulator’s architecture, memory model, types, and
control-flow mechanics into a single reference.

Architecture in one view
------------------------

OpenInfer is an explicit execution system: you describe computation and control
flow in a graph, and the runtime executes it deterministically.

Core idea:

1. Models live in a single `.oinf` package (weights, sizes, metadata).
2. Developers write execution logic with the `graph!` DSL.
3. The DSL produces a symbolic `Graph` (blocks + nodes).
4. The runtime executes, traces, and validates the graph on CPU or Vulkan.

.. code-block:: text

   Model package (.oinf)
           |
           v
       graph! DSL  -->  Graph (blocks + nodes)
                            |
                            +--> Simulator / Executor (CPU | Vulkan)
                            +--> Trace + JSON serialization
                            +--> Analyzer / Synthesizer (planned codegen)

Subsystems
----------

- **Graph**: blocks, nodes, and variable declarations.
- **Runtime**: validation, execution, tracing, and scheduling.
- **Tensor**: dtype system, shapes, and storage utilities.
- **Ops**: kernel registry and dispatch across CPU and Vulkan.
- **OINF**: model packaging and lazy loading of weights.
- **DSL**: procedural macro for building graphs in Rust.

Graph and runtime basics
------------------------

- A `Graph` contains named `Block` values.
- A `Block` contains an ordered list of `Node` values.
- `NodeKind` variants represent ops, control flow, cache access, and sync.

The graph is a control-flow graph, not just a list of ops. Blocks can jump with
`branch`, repeat with `loop`, and synchronize with `yield`/`await`. Execution
follows graph order; there is no hidden scheduler that reorders nodes.

Memory model
------------

Memory is explicit in the DSL. Variables are declared in sections:

- `dynamic`: provided at runtime and cleared each step.
- `volatile`: mutable during execution and reset each step.
- `constant`: immutable values loaded from the model package.
- `persistent`: mutable state that survives across steps.

The executor API bridges host data and graph variables. Use
`insert_executor!` to populate `dynamic` inputs and `fetch_executor!` to read
results or state.

Tables and cache operations
---------------------------

Persistent variables can be treated as cache tables with explicit operations:

- `cache.read`
- `cache.write`
- `cache.increment` / `cache.decrement`
- `cache.reset`

Tables are indexed by symbolic variables (e.g., `layer`, `time`) declared in the
DSL. You control index updates explicitly, which makes cache growth and ordering
deterministic and traceable.

Auto-dim tables grow when indices increase:

.. code-block:: rust

   persistent {
     rows: i32 @init(0);
     cols: i32 @init(0);
     M(r, c): f16[D, H] @auto_dim(r, c);
   }

   block entry {
     cache.increment 3 rows;
     cache.increment 5 cols;
     cache.read M[rows, cols] >> out;
     return;
   }

Types and dtypes
----------------

Universal types:

- f32 / f64
- i8 / i16 / i32 / i64
- u8 / u16 / u32 / u64
- bool
- bitset

Special tensor types:

- f8 / f16 / bf16
- i1 / i2 / i4
- u1 / u2 / u4
- t1 / t2 (reserved)

Packed types remain packed in storage to reduce bandwidth. Vulkan support for
`i64/u64` and `f64` depends on `shader_int64` and `shader_float64`; unsupported
types fall back to CPU with a warning.

Control flow and ordering
-------------------------

Blocks are the unit of control flow. A block can `branch` to another block or
repeat with `loop`. Ordering is explicit:

- `barrier` prevents reordering across a boundary.
- `dep after(x) before(y)` enforces ordering without data edges.

`yield` and `await` coordinate streaming or async-style pipelines. Every await
must have a matching yield in the graph.

Tracing and serialization
-------------------------

Tracing captures node index, block name, op name, and timing. Enable it with:

.. code-block:: rust

   let sim = Simulator::new(&model, &g, Device::Cpu)?
       .with_trace()
       .with_timer();

Graphs can be serialized to JSON:

.. code-block:: rust

   let json = GraphSerialize::json(&g)?;
   std::fs::write("graph.json", serde_json::to_string_pretty(&json)?)?;

and deserialized back:

.. code-block:: rust

   let value = serde_json::from_str(&graph_txt)?;
   let g = GraphDeserialize::from_json(value)?;
