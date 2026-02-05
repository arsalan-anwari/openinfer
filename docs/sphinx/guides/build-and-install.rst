Build and Install
=================

This guide covers common build targets, feature flags, and Vulkan setup.

CPU-only build
--------------

.. code-block:: bash

   cargo build

To build a specific crate:

.. code-block:: bash

   cargo build -p openinfer
   cargo build -p openinfer-dsl

Vulkan build
------------

Vulkan ops are gated behind a feature flag:

.. code-block:: bash

   cargo build --features vulkan

If you plan to run the Vulkan examples, enable the feature on `cargo run` too:

.. code-block:: bash

   cargo run --example streaming_pipeline --features vulkan -- --target=vulkan

If Vulkan is enabled, make sure a Vulkan SDK is installed and your system
exposes a compatible device.

Troubleshooting Vulkan setup:

- Verify `vulkaninfo` is available.
- Check that a device is listed under `GPU id`.
- For headless setups, verify a compute-capable device is visible.

Environment variables
---------------------

OpenInfer uses a few environment variables for tracing:

- `OPENINFER_TRACE`: `1` or `full` to enable CPU tracing.
- `OPENINFER_VULKAN_TRACE`: `1` or `full` to enable Vulkan tracing.

Example:

.. code-block:: bash

   OPENINFER_TRACE=1 cargo run --example mlp_regression
   OPENINFER_TRACE=full cargo run --example mlp_regression

Trace events are accumulated in memory and can be read via `Executor::trace()`
if you are embedding OpenInfer in Rust.

SPIR-V artifacts
----------------

Vulkan shaders are compiled into SPIR-V binaries that are embedded at build
time. When you add new shader sources, rebuild the workspace so the embedded
map is refreshed.

If you are iterating on shader sources:

.. code-block:: bash

   cargo clean-spv -p openinfer
   cargo clean -p openinfer
   cargo build-spv -p openinfer
   cargo build --features vulkan

Documentation build
-------------------

Build Sphinx and Rustdoc together:

.. code-block:: bash

   ./scripts/build_docs.sh

This script creates a local `.venv` and installs Sphinx dependencies from
`docs/sphinx/requirements.txt`.

SPIR-V and opspec generation
----------------------------

The project defines cargo aliases to generate SPIR-V shader blobs and opspec
artifacts from `ops.json`:

.. code-block:: bash

   cargo build-spv
   cargo build-opspec

These are most useful when you modify Vulkan shader sources or update `ops.json`
and want the generated artifacts refreshed. The aliases are defined in
`.cargo/config.toml`.

Platform notes
--------------

- Linux: Vulkan is supported on most discrete and integrated GPUs.
- macOS: Vulkan requires MoltenVK and may be limited.
- Windows: Ensure a recent Vulkan SDK is installed.
