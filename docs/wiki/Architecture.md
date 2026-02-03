# Architecture (Overview)

This is a surface‑level view of how the library is structured and how data
flows through it.

## High‑level flow

1. **Load model**: `ModelLoader` opens a `.oinf` file and reads headers.
2. **Build graph**: `graph! {}` expands into a `Graph` data structure.
3. **Validate**: the simulator checks dtypes, shapes, and op attributes.
4. **Execute**: the executor runs nodes on CPU or Vulkan.
5. **Inspect**: optional trace/timing output is available.

## Codebase layout (user‑level)

- `openinfer/` — main runtime crate (DSL integration, execution, backends)
- `openinfer-dsl/` — procedural macro that builds graphs
- `openinfer-generator/` — utilities for Vulkan shaders and build artifacts
- `openinfer-oinf/` — Python tools for creating and validating `.oinf` files
- `examples/` — Rust + Python examples
- `res/models/` — example `.oinf` model files

## Execution model

- Graph nodes execute in explicit order within blocks.
- Control flow creates a graph of blocks (not just a flat list).
- Persistent memory is stored across steps and updated explicitly via cache ops.
- CPU is the reference backend; Vulkan is optional for GPU execution.

This design favors explicitness and inspectability over opaque scheduling.
