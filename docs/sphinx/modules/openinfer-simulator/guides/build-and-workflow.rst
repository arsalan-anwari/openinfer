Build and Workflow
==================

This page consolidates build, scripts, testing, troubleshooting, and examples.

Build and install
-----------------

CPU-only build:

.. code-block:: bash

   cargo build

Vulkan build:

.. code-block:: bash

   cargo build --features vulkan

If you plan to run Vulkan examples:

.. code-block:: bash

   cargo run --example streaming_pipeline --features vulkan -- --target=vulkan

Environment variables for tracing:

- `OPENINFER_TRACE`: `1` or `full` for CPU tracing.
- `OPENINFER_VULKAN_TRACE`: `1` or `full` for Vulkan tracing.

Documentation build
-------------------

.. code-block:: bash

   ./scripts/build_docs.sh

The script creates a local `.venv`, builds Rustdoc, and copies Rustdoc into
`docs/sphinx/out/api/rustdoc` for Sphinx links.

Scripts and workflows
---------------------

Run examples (generates `.oinf` via Python first):

.. code-block:: bash

   ./scripts/run_examples.sh --list
   ./scripts/run_examples.sh --target=cpu
   ./scripts/run_examples.sh --target=vulkan --features=vulkan
   ./scripts/run_examples.sh --example-filter kv_cache_decode

Sync models from `openinfer-oinf` into the simulator:

.. code-block:: bash

   ./scripts/sync_models.sh

Setup and build (repo workflow):

.. code-block:: bash

   ./scripts/setup_all.sh
   ./scripts/build_all.sh

Testing
-------

Rust tests:

.. code-block:: bash

   cargo test --manifest-path openinfer-simulator/Cargo.toml

Python `.oinf` verification:

.. code-block:: bash

   python openinfer-oinf/tests/run_oinf_tests.py

Baselines live under:

``openinfer-simulator/tests/ops/baseline/data``

Examples
--------

Rust examples (`openinfer-simulator/examples`):

- `mlp_regression`: basic MLP forward pass.
- `linear_attention`: loop-based linear attention.
- `quantized_linear`: i4 quantized matmul.
- `moe_routing`: MoE routing with `branch`.
- `residual_mlp_stack`: residual stack with patterned weights.
- `stability_guard`: `is_finite` guard with fallback.
- `streaming_pipeline`: `yield`/`await` pipeline.
- `online_weight_update`: persistent updates via cache ops.
- `kv_cache_decode`: fixed-size KV cache read.
- `cache_window_slice`: cache window slicing.

Python `.oinf` examples (`openinfer-oinf/examples`) mirror the Rust list.

Troubleshooting
---------------

Build failures:

- Missing Rust toolchain: install via `rustup`.
- Missing Python deps: run `pip install -r requirements.txt`.
- Vulkan build errors: ensure `--features vulkan` and a Vulkan SDK.

Runtime validation errors:

- Graph variable dims do not match `.oinf` sizevars.
- DSL variable name conflicts with model variable of a different dtype.

Vulkan fallback:

- If a GPU lacks `shader_int64` or `shader_float64`, Vulkan ops fall back to CPU.
