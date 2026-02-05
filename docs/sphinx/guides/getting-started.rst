Getting Started
===============

This guide walks through a minimal build and run to confirm your environment is
working, then points you to deeper module docs.

Prerequisites
-------------

- Rust toolchain (stable): `rustc`, `cargo`.
- Python 3.10+ for `.oinf` tooling and examples.
- Vulkan SDK only if you plan to run GPU ops (`--features vulkan`).

Optional tools:

- `vulkaninfo` to verify Vulkan device setup.

Repository layout snapshot
--------------------------

At a high level:

- `openinfer/`: runtime, graph, tensors, CPU/Vulkan ops.
- `openinfer-dsl/`: `graph!` macro and parsing.
- `openinfer-oinf/`: Python tools for `.oinf`.
- `examples/`: runnable example programs (Rust + Python).
- `scripts/`: automation for examples, tests, docs, wiki sync.

Quick build
-----------

From the repo root:

.. code-block:: bash

   cargo build

If you want Vulkan enabled:

.. code-block:: bash

   cargo build --features vulkan

Optional generator targets
--------------------------

If you change `ops.json` or Vulkan shaders, regenerate artifacts:

.. code-block:: bash

   cargo build-opspec
   cargo build-spv

Optional: build examples only

.. code-block:: bash

   cargo build --examples

Run a minimal example
---------------------

The examples directory contains end-to-end graph execution. This example loads
an `.oinf` file and executes a small MLP:

.. code-block:: bash

   cargo run --example mlp_regression

If the example fails, try a CPU-only build to isolate GPU issues:

.. code-block:: bash

   cargo run --example mlp_regression --no-default-features

If the build succeeds and the example prints output, your toolchain is ready.

Minimal graph walkthrough
-------------------------

OpenInfer graphs are defined with the `graph!` DSL and executed with a
`Simulator` and `Executor`. The snippet below illustrates the flow:

.. code-block:: rust

   let model = ModelLoader::open("res/models/mlp_regression.oinf")?;
   let g = graph! {
       dynamic { x: f32[B, D]; }
      constant { w1: f32[D, H]; b1: f32[H]; w2: f32[H, O]; b2: f32[O]; }
       volatile { h: f32[B, H]; y: f32[B, O]; }
       block entry {
           op matmul(x, w1) >> h;
           op add(h, b1) >> h;
           op relu(h, alpha=0.0) >> h;
           op matmul(h, w2) >> y;
           op add(y, b2) >> y;
           return;
       }
   };
   let sim = Simulator::new(&model, &g, Device::Cpu)?;
   let mut exec = sim.make_executor()?;
   exec.step()?;

Key ideas:

- `.oinf` files hold constants, sizes, and metadata.
- The DSL defines control flow and ops explicitly.
- Execution is deterministic and traceable.

Executor macros
---------------

Macros bridge user data and the executor:

.. code-block:: rust

   insert_executor!(exec, { x: input });
   exec.step()?;
   fetch_executor!(exec, { y: Tensor<f32> });

The `try_*` variants return `Result` instead of panicking.

Generate models for examples (Python)
-------------------------------------

Rust examples consume `.oinf` models, which can be generated with Python:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python examples/openinfer-oinf/mlp_regression_oinf.py
   python openinfer-oinf/verify_oinf.py res/models/mlp_regression.oinf

Run full example pipeline
-------------------------

The `scripts/run_examples.sh` script runs the Python model generator and then
the Rust example.

.. code-block:: bash

   ./scripts/run_examples.sh --list
   ./scripts/run_examples.sh --target=cpu
   ./scripts/run_examples.sh --target=vulkan --features=vulkan

Quick test pass
---------------

Run a narrow test to confirm the harness works:

.. code-block:: bash

   ./scripts/run_tests.sh --test-filter openinfer::graph::graph_simple

Next steps
----------

- Read `Architecture Overview` for the mental model.
- Explore module-specific docs under `Modules`.
