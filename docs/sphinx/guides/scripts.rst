Scripts and Workflows
=====================

This guide documents the scripts under `scripts/` and the workflows they enable.

Build documentation
-------------------

`build_docs.sh` creates Sphinx output and copies Rustdoc into the Sphinx output
tree so links work.

.. code-block:: bash

   ./scripts/build_docs.sh

What it does:

- creates/uses `.venv`
- installs Sphinx deps
- generates ops catalog from `ops.json`
- builds Rustdoc (`cargo doc --workspace --no-deps`)
- builds Sphinx HTML
- copies Rustdoc into `docs/sphinx/out/api/rustdoc`

Common issues:

- If `sphinx-build` fails, delete `.venv` and re-run.
- If Rustdoc links are broken, verify the `out/api/rustdoc` folder exists.

Run examples
------------

`run_examples.sh` generates Python `.oinf` models and then runs Rust examples.

.. code-block:: bash

   ./scripts/run_examples.sh --list
   ./scripts/run_examples.sh --target=cpu
   ./scripts/run_examples.sh --target=vulkan --features=vulkan
   ./scripts/run_examples.sh --example-filter kv_cache_decode

Workflow details:

1. Python example script generates `.oinf` files under `res/models`.
2. Rust example loads the model via `ModelLoader::open`.
3. Execution runs on CPU or Vulkan depending on `--target`.

Tip: use `--example-filter` to isolate a single example when iterating on a bug.

Run tests
---------

`run_tests.sh` generates baselines and runs Rust + Python tests.

.. code-block:: bash

   ./scripts/run_tests.sh --target=cpu
   ./scripts/run_tests.sh --target=vulkan --features=vulkan
   ./scripts/run_tests.sh --test-filter openinfer::ops_misc
   ./scripts/run_tests.sh --test-filter openinfer-dsl::parse_tests
   ./scripts/run_tests.sh --test-filter openinfer-oinf::test_common.TestCommon.test_align_up

Baseline generation:

- `gen_ops_baseline.py` and `gen_graph_baseline.py` create reference data.
- These are run automatically unless you filter to non-openinfer test modules.

Test targets:

- `--target=cpu` runs CPU kernels only.
- `--target=vulkan` runs the Vulkan backend if enabled.

Update wiki
-----------

`update_wiki.sh` syncs `docs/wiki/` to the GitHub wiki repo.

.. code-block:: bash

   ./scripts/update_wiki.sh
   ./scripts/update_wiki.sh --output-dir /path/to/parent

This script clones `<repo>.wiki.git` and uses `rsync` to mirror `docs/wiki`.

Dry-run workflow:

.. code-block:: bash

   ./scripts/update_wiki.sh --output-dir /tmp/wiki-preview

Workflow diagram
----------------

.. mermaid::

   sequenceDiagram
     participant Dev
     participant Scripts
     participant Docs
     Dev->>Scripts: build_docs.sh
     Scripts->>Docs: generate ops docs
     Scripts->>Docs: build Rustdoc
     Scripts->>Docs: build Sphinx HTML
