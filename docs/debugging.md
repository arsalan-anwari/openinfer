Debugging Guide
===============

This guide describes practical ways to debug OpenInfer across CPU, SIMD, and Vulkan
backends, as well as graph tracing and model validation. It is intended for local
workflows and can be used on any platform with the Rust toolchain installed.

General Tips
------------
- Build with debug symbols for readable backtraces:
  `cargo build`
- Capture a backtrace on crash:
  `RUST_BACKTRACE=1 <command>`
- Prefer running from the repo root so relative paths resolve correctly.
- If you suspect stale build artifacts, use a clean build:
  `cargo clean`

Graph and Execution Traces
--------------------------
- Enable execution tracing to see op order, block names, and timings:
  `OPENINFER_TRACE=1 cargo run --example <name>`
- Use the Rust examples to validate branching and per-op behavior:
  - `examples/rust/branching_good.rs`
  - `examples/rust/branching_bad.rs`
  - `examples/rust/ops_matrix.rs`
  - `examples/rust/yield.rs`
- The trace output includes block names so you can confirm control-flow paths.

Model and DSL Validation
------------------------
- If a graph fails to parse or validate, check the DSL validation errors.
- The validator enforces:
  - Op attribute types (e.g., `fill` value dtype).
  - Input/output dtype consistency.
  - Broadcast policy constraints.
- If the error mentions `OpAttrs` or unknown variables, inspect the generated
  graph in the example source to ensure all names match.

CPU Backend Debugging
---------------------
- The CPU backend is the reference implementation for correctness.
- To isolate CPU output, run without GPU features:
  `cargo run --example ops_matrix`
- AVX/AVX2 variants can be tested explicitly with:
  - `cargo run --example ops_matrix -- --target=avx`
  - `cargo run --example ops_matrix -- --target=avx2`
- If SIMD output diverges, compare against the CPU reference printed by
  `ops_matrix.rs`.

Vulkan Backend Debugging
------------------------
- Enable Vulkan tracing for device setup and dispatch logging:
  `OPENINFER_VULKAN_TRACE=1 cargo run --example ops_matrix --features vulkan -- --target=vulkan`
- The trace prints:
  - Vulkan init steps (instance/device/pools/layouts).
  - Per-op dispatch details (op, dtype, entry, push constants, descriptor set).
- If you hit a kernel lookup error, verify that:
  - The dtype is supported on Vulkan (see `docs/ops.md`).
  - The op has a shader entry in `openinfer/src/ops/vulkan/shaders.json`.
- If a shader fails to compile, check the Slang file path printed by the build
  script and the specific type reported by `slangc`.

Descriptor Set Limits
---------------------
- Vulkan uses a fixed number of descriptor sets per pipeline layout.
- The runtime may group dtypes by shader set; exceeding the configured limit
  will raise an error.
- If you need more dtypes in a single run, consider splitting workloads or
  reducing the number of dtypes tested in one graph.

Op-Specific Debugging
---------------------
- `is_finite`: returns a bool scalar; for non-float types, it always returns true.
- `fill`: fills output with a typed literal. If the value type does not match
  the input dtype, validation will reject it.
- `matmul`: requires compatible inner dimensions; mismatches will error.

Common Failure Modes
--------------------
- Unknown op or attribute: check the DSL input and `openinfer/src/graph.rs`.
- Kernel not found: check registry wiring and dtype support in `docs/ops.md`.
- Segfaults on Vulkan: run with `OPENINFER_VULKAN_TRACE=1` and reduce the
  number of dtypes or ops in a single graph to isolate.

Useful Files
------------
- Backend registries:
  - `openinfer/src/ops/cpu/registry.rs`
  - `openinfer/src/ops/vulkan/registry.rs`
- Vulkan runtime:
  - `openinfer/src/backend/vulkan/mod.rs`
  - `openinfer/src/backend/vulkan/runtime.rs`
- Graph and DSL:
  - `openinfer/src/graph.rs`
  - `openinfer-dsl/src/validation/ops/`
