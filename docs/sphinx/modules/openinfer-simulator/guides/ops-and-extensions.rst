Ops and Extensions
==================

This page combines the ops catalog overview with the end-to-end workflow for
adding new operations.

Ops catalog
-----------

Operations are defined in `ops.json` and loaded at runtime. The generated
catalog lives under:

- :doc:`../ops/index`

What ops define
------------------

- Name and arity.
- Attributes and attribute types.
- Broadcast/in-place/accumulate support.
- DType coverage for CPU and Vulkan.
- Output dtype inference rules.

Regeneration
---------------

The ops catalog is generated from `ops.json` by:

.. code-block:: bash

   python docs/sphinx/generate_ops_docs.py

This is run automatically by `./scripts/build_docs.sh` before Sphinx builds.

Adding a new op (summary)
-------------------------

1. Add schema in `ops.json`.
2. Implement CPU kernel(s).
3. Add Vulkan shader (optional).
4. Add tests and baseline data.
5. Regenerate ops docs.

End-to-end steps (expanded)
---------------------------

**Step 1: Define the op in `ops.json`.**

- Add `name`, `inputs/outputs`, `attrs`, capability flags, dtype rules.
- Reference shared dtype sets.
- Add per-device entries for CPU/Vulkan when supported.

After edits, regenerate docs:

.. code-block:: bash

   python docs/sphinx/generate_ops_docs.py

**Step 2: Wire the op in the runtime.**

- Add the op name to `OpKind` in `openinfer-simulator/src/graph/types.rs`.
- Ensure attributes are parsed and validated.
- Rebuild the crate to refresh the embedded registry.

**Step 3: Implement the CPU kernel.**

- Add kernel file(s) under `openinfer-simulator/src/ops/cpu/`.
- Implement per-dtype kernels.
- Register them in the CPU registry.

**Step 4: Implement the Vulkan shader (optional).**

- Add `.slang` shader sources under `openinfer-simulator/src/ops/vulkan/`.
- Register shader entrypoints in the Vulkan registry.
- Ensure `ops.json` lists the shader files and SPIR-V output dirs.

**Step 5: Update validation rules.**

- Attribute validation (types and ranges).
- Dtype compatibility checks.
- Output dtype inference rules.

**Step 6: Add tests and baselines.**

- CPU correctness per dtype.
- Vulkan correctness (if supported).
- In-place and accumulation modes (if supported).
- Regenerate baselines if required.

Common pitfalls
---------------

- Mismatched op names across DSL, registry, and schema.
- Missing dtype variants in registries.
- Attribute type mismatches (e.g., int vs float).
- Packed dtype handling on CPU/Vulkan.

Where to go next
----------------

- :doc:`../ops/index` for the full per-op catalog.
- The tutorials for hands-on examples.
