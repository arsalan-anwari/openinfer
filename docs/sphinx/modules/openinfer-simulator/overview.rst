Overview
========

The `openinfer` crate (located at `openinfer-simulator/`) defines the core
graph model, tensor types, and host-side simulation runtime. It validates graph
correctness, scheduling logic, and memory layouts without relying on target
devices.

Topics
------

- Public entry points and primary types
- Role within the workspace

Core entry points
-----------------

- `Graph`, `Block`, `Node`: graph structure and control flow.
- `Tensor<T>`, `TensorValue`: data containers and dtype handling.
- `Simulator`, `Executor`: validation and host-side execution.
- `ModelLoader`: lazy `.oinf` loader.

How it fits
-----------

The simulator is the grounding for the workflow: the DSL builds graphs that are
validated and executed here, and the `.oinf` tooling provides model packages
loaded during simulation. The synthesizer will later lower the same graphs into
device-specific code for edge targets.


