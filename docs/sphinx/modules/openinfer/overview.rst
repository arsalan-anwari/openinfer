Overview
========

The `openinfer` crate defines the core graph model, tensor types, and runtime
execution APIs.

Topics
------

- Public entry points and primary types
- Role within the workspace

Core entry points
-----------------

- `Graph`, `Block`, `Node`: graph structure and control flow.
- `Tensor<T>`, `TensorValue`: data containers and dtype handling.
- `Simulator`, `Executor`: validation and execution.
- `ModelLoader`: lazy `.oinf` loader.

How it fits
-----------

The `openinfer` crate is the runtime engine for the workspace. The DSL crate
builds graphs consumed here, and the `.oinf` tooling provides model packages
that are loaded at runtime.
