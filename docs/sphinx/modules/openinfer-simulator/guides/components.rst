Components Overview
===================

OpenInfer is organized into a few core subsystems that work together to execute
graphs deterministically.

Subsystems
----------

- **Graph**: data structures representing blocks, nodes, and variables.
- **Runtime**: validation, execution, tracing, and scheduling.
- **Tensor**: dtype system, shapes, and storage utilities.
- **Ops**: kernel registry and dispatch across CPU and Vulkan.
- **OINF**: model packaging and lazy loading of weights.
- **DSL**: procedural macro for building graphs in Rust.

Graph subsystem
---------------

The graph subsystem defines the runtime IR: nodes, blocks, and variable
declarations. A graph is built either via the `graph!` DSL or programmatically
using `Graph::add_var`, `Graph::add_block`, and `Graph::add_node`. Nodes encode
both computation and control flow, which makes the graph suitable for tracing,
debugging, and future optimization passes.

Runtime subsystem
-----------------

The runtime owns validation, execution, tracing, and scheduling:

- `Simulator` validates graph/model compatibility and prepares execution.
- `Executor` runs the graph, tracks state, and exposes fetch APIs.
- `TraceEvent` records node timing and metadata.

Ops subsystem
-------------

The ops subsystem loads op schemas from `ops.json` and dispatches kernels based
on dtype, device, and capability flags (broadcast/inplace/accumulate). CPU and
Vulkan backends are registered independently, but follow the same schema.

Tensor subsystem
----------------

Tensors provide storage and metadata. `Tensor<T>` owns a flat buffer and carries
shape and stride information. `TensorValue` wraps multiple concrete dtypes to
enable runtime dispatch without dynamic typing.

OINF subsystem
--------------

The `.oinf` format is a deterministic binary container for model data. The
loader memory-maps this file and lazily loads tensor payloads only when needed.

DSL subsystem
-------------

The DSL is a compile-time macro that parses a small grammar into a `Graph`. It
is explicit by design and avoids implicit optimizations or hidden control flow.

System diagram
--------------

.. mermaid::

   flowchart TD
     DSL["Graph DSL"] --> Graph["Graph IR"]
     OINF["OINF Model"] --> Loader["ModelLoader"]
     Graph --> Simulator["Simulator"]
     Loader --> Simulator
     Simulator --> Executor["Executor"]
     Executor --> Ops["Kernel Dispatch"]
     Ops --> CPU["CPU Kernels"]
     Ops --> VK["Vulkan Kernels"]
     Executor --> Trace["Trace Output"]
