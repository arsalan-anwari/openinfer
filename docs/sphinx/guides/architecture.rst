Architecture Overview
=====================

OpenInfer is an explicit execution system: you describe the computation and control flow in a graph, and the runtime executes it deterministically.

Core idea
---------

1. Models live in a single `.oinf` package (weights, sizes, metadata).
2. Developers write execution logic with the `graph!` DSL.
3. The DSL produces a symbolic `Graph` (blocks + nodes).
4. The runtime executes, traces, and validates the graph on CPU or Vulkan.

High-level flow
---------------

.. code-block:: text

   Model package (.oinf)
           |
           v
       graph! DSL  -->  Graph (blocks + nodes)
                            |
                            +--> Simulator / Executor (CPU | Vulkan)
                            +--> Trace + JSON serialization
                            +--> Analyzer / Synthesizer (planned)

The system is designed so the graph is the single source of truth. There is no implicit scheduling or hidden optimizer layer. This makes the runtime easier to reason about and gives developers precise control when building new workloads.

System diagram
--------------

.. mermaid::

   flowchart TD
     A[graph! DSL] --> B[Graph IR]
     C[OINF Model] --> B
     B --> D[Simulator]
     D --> E[Executor]
     E --> F[CPU Ops]
     E --> G[Vulkan Ops]
     E --> H[Trace + JSON]

Key concepts
------------

Graph structure
~~~~~~~~~~~~~~~

- A `Graph` contains named `Block` values.
- A `Block` contains an ordered list of `Node` values.
- `NodeKind` variants represent ops, control flow, cache access, and sync.

The graph is a control-flow graph, not just a list of ops. Blocks can jump with `branch`, repeat with `loop`, and synchronize with `yield`/`await`. This keeps control flow explicit and analyzable for debugging and future optimization.

Graph lifecycle
~~~~~~~~~~~~~~~

1. The DSL or API builds a `Graph` with explicit variable declarations.
2. The `Simulator` checks the graph against the model and run config.
3. The `Executor` runs each block and node, updating runtime state.
4. Optional tracing serializes the execution to JSON.

Runtime execution
~~~~~~~~~~~~~~~~~

- `Simulator` validates the graph against the model and config.
- `Executor` runs nodes in order, handling branches and loops.
- Traces are emitted when tracing is enabled.

Validation includes sizevar resolution, dtype compatibility, and attribute type checking. Execution is deterministic, and tracing captures node timing and metadata for post-mortem analysis.

DSL to graph
~~~~~~~~~~~~

The `graph!` macro parses a compact grammar and expands into Rust code that constructs a `Graph`. It maps memory declarations to `VarDecl` values and block nodes to `NodeKind` variants. This keeps the runtime logic in `openinfer` independent from the DSL parser.

DSL structure summary:

- Memory sections: `dynamic`, `volatile`, `constant`, `persistent`
- Blocks: `block entry { ... }`
- Nodes: `assign`, `op`, cache ops, `branch`, `loop`, `yield`, `await`

Example DSL snippet:

.. code-block:: rust

   let g = graph! {
     dynamic {
       x: f32[B, D];
     }
     constant {
       W: f32[D, D];
     }
     volatile {
       y: f32[B, D];
     }

     block entry {
       op matmul(x, W) >> y;
       return;
     }
   };

Tensor system
~~~~~~~~~~~~~

- `Tensor<T>` holds data, shape, and strides.
- `TensorValue` wraps concrete tensors across all supported dtypes.
- Packed types (i1/i2/i4/u1/u2/u4/t1/t2) remain packed in storage.

Tensor values flow through nodes. Each op declares input and output dtypes, and the runtime verifies the actual tensors satisfy those contracts.

Model loading
~~~~~~~~~~~~~

`ModelLoader::open` memory-maps the `.oinf` file, validates header tables, and only loads tensor payloads when needed. This makes model parameters lazy and keeps large models manageable.

Lazy loading path:

1. Executor requests a tensor by name.
2. Loader finds offsets in the `.oinf` index.
3. Only the requested byte range is read.
4. Data is cached for subsequent uses.

The cache uses the tensor name as a stable key so repeated reads are fast and deterministic across blocks.

Op selection
~~~~~~~~~~~~

Op dispatch uses the registry from `ops.json`:

- Inputs determine kernel dtype variants.
- Attributes determine output dtype (including accumulation).
- Device selection chooses CPU or Vulkan implementation.

Tracing
~~~~~~~

Trace events include node index, UUID, op name, block name, and timing. This is used for debugging correctness and performance.

Trace workflow:

.. code-block:: bash

   OPENINFER_TRACE=full cargo run --example kv_cache_decode

The JSON trace is a structured log that can be ingested by downstream tooling.

Control flow and cache
~~~~~~~~~~~~~~~~~~~~~~

- `branch`, `loop`, `yield`, `await`, `barrier`, and `dep` encode explicit execution order.
- Cache ops provide explicit persistent storage access.

Where to go next
----------------

- `Types` for dtype details and backend behavior.
- `Control Flow` for blocks, branches, and yields.
- `Modules/openinfer` for runtime internals.
