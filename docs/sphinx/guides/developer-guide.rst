Developer Guide
===============

This guide covers repository structure, development workflow, and debugging.

Workspace layout
----------------

- `openinfer/`: core runtime, graph, tensor, and ops.
- `openinfer-dsl/`: `graph!` procedural macro and parsing.
- `openinfer-oinf/`: Python tooling for `.oinf` encoding and verification.
- `openinfer-generator/`: code generation utilities (ops, SPIR-V).
- `docs/`: design and reference documentation.
- `scripts/`: build and verification scripts.
- `res/`: resource files for examples and tests.
- `examples/`: example models and scripts.
- `tests/`: unit and integration tests.

Development workflow
--------------------

1. Make code changes in the relevant crate.
2. Run targeted tests (Rust or Python).
3. Run an example for end-to-end validation.

Tracing and timing
------------------

Tracing is opt-in via the simulator:

.. code-block:: rust

   let sim = Simulator::new(&model, &graph, Device::Cpu)?
       .with_trace()
       .with_timer();
   let mut exec = sim.make_executor()?;
   exec.step()?;

Tracing is stored in memory and can be retrieved via `Executor::trace()`.

Testing strategy
----------------

- Rust unit/integration tests live under `tests/`.
- Baseline data lives under `tests/openinfer/ops/baseline/data`.
- Python verification scripts live under `tests/openinfer-oinf`.

Code style notes
----------------

- The DSL favors explicitness over implicit conversions.
- Validation errors are expected to be precise and deterministic.

Macros and helpers
------------------

Use the executor macros for quick scripts and tests:

.. code-block:: rust

   insert_executor!(exec, { x: input })?;
   fetch_executor!(exec, { y: Tensor<f32> })?;

The `try_*` variants return `Result` and are preferred in library code.
