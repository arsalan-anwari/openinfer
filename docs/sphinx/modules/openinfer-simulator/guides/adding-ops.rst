Adding a New Op (End-to-End)
============================

This chapter is a comprehensive, end-to-end guide for adding a new operation to
OpenInfer. It covers the full lifecycle from schema definition in `ops.json` to
CPU kernels, Vulkan shaders, registry wiring, tests, and documentation. The goal
is not just to make the change work, but to make it *maintainable* and *traceable*
for the next developer who touches it.

Because the system is explicit by design, adding an op is a structured process.
You define the op schema, add implementations, register them, and then verify
behavior through tests and docs. This is more steps than a dynamic runtime, but
it gives you confidence and reproducibility.

Before you start, review the `Ops` guide and the ops catalog generated in
`docs/sphinx/ops/`. Those pages list the existing ops, dtypes, and capabilities.
You should align your new op with those conventions so the documentation stays
consistent.

Step 1: Define the op in `ops.json`
--------------------------------------

The op schema is the single source of truth for the op’s name, inputs, outputs,
attributes, dtype coverage, and capability flags. The runtime reads `ops.json`
at build time and uses it to validate and dispatch kernels.

A typical entry includes:

- `name` and `kind`: name in the DSL and enum tag (`OpKind`) in Rust.
- `inputs` / `outputs`: arity and counts.
- `attrs`: attribute list (by name, referencing `attr_defs`).
- `broadcast`, `inplace`, `accumulate`: capability flags as strings.
- `type_rule`: output dtype rule (e.g. same-as-input).
- `dtype_support_ref`: reference into the shared `dtype_sets`.
- `devices`: per-device configuration (CPU enabled, Vulkan shader info).

Example (matching actual structure):

.. code-block:: json

   {
     "name": "add",
     "kind": "add",
     "inputs": { "arity": "fixed", "count": 2 },
     "outputs": { "arity": "fixed", "count": 1 },
     "attrs": ["acc"],
     "broadcast": "allow",
     "inplace": "allow",
     "accumulate": "allow",
     "type_rule": { "kind": "same_as_input", "index": 0 },
     "dtype_support_ref": "ADD_DTYPE_SUPPORT",
     "devices": {
       "cpu": { "enabled": true },
       "vulkan": {
         "shader_dir": "src/ops/vulkan/arithmetic/add/shaders",
         "shader_files": ["normal.slang", "accumulate.slang", "packed.slang"],
         "spv_dir": "src/ops/vulkan/arithmetic/add/bin"
       }
     }
   }

When you add the entry, keep the op name lowercase and consistent with the
existing list. Avoid special characters; the name is used in filenames and in
shader entrypoints.

After editing `ops.json`, regenerate the ops catalog:

.. code-block:: bash

   python docs/sphinx/generate_ops_docs.py

This keeps the Sphinx docs in sync and verifies that the schema is still valid.

Step 2: Add the op spec and enum wiring
---------------------------------------

Most ops are represented in a Rust enum or registry mapping. The op name must
be recognized by the runtime so that `NodeKind::Op` can resolve the proper
kernel. The exact file depends on the structure of `openinfer-simulator/src/ops/`.

At a minimum you will:

1. Add the op name to `OpKind` in `openinfer-simulator/src/graph/types.rs`.
2. Ensure the op attributes are parsed and validated.
3. Wire the op to the kernel registry.

If the op has attributes, confirm that they are represented in the `OpAttrs`
structure and validated in the op validation layer. This ensures incorrect
attributes are caught at simulation time.

The op schemas are parsed from `ops.json` in `openinfer-simulator/src/op_defs.rs` at
compile time. If you update the schema, make sure you rebuild the crate so the
embedded registry is refreshed.

If the op spec or schema artifacts are generated, refresh them using:

.. code-block:: bash

   cargo build-opspec

Step 3: Implement the CPU kernel
--------------------------------

CPU kernels live under `openinfer-simulator/src/ops/cpu/`. The typical pattern is:

1. Add a kernel implementation file for the op.
2. Implement the per-dtype kernel functions.
3. Register the kernel in the CPU registry.

Example skeleton:

.. code-block:: rust

   pub fn swish_f32_normal(input: &Tensor<f32>, beta: f32, out: &mut Tensor<f32>) {
       for (x, y) in input.iter().zip(out.iter_mut()) {
           let v = *x;
           *y = v / (1.0 + (-beta * v).exp());
       }
   }

If you implement multiple dtypes, follow the conventions used by existing ops:

- Use `f32` internally for `f16`/`bf16` when needed.
- Keep packed types in packed form unless the op requires unpacking.
- Avoid allocations inside the kernel; use output buffers passed in.
- Look at references of other ops to see which function prototypes to implement or look at the generated code of the `kernel.rs` file for the list of functions to implement.

If the op supports in-place or accumulation modes, implement those variants as
separate functions or as modes inside the kernel. The registry will dispatch the
correct mode based on the op flags and graph context.

Step 4: Implement the Vulkan shader
-----------------------------------

Vulkan ops live under `openinfer-simulator/src/ops/vulkan/` and are usually paired with a
`.slang` shader file. The flow is:

1. Add a new shader under the appropriate category (e.g. `arithmetic/`).
2. Define the entrypoint name and its buffer bindings.
3. Implement the compute logic, respecting the dtype and packing rules.
4. Register the shader in the Vulkan registry.

Shader locations and output SPIR-V paths are recorded in `ops.json` under
`devices.vulkan`. The `cargo build-spv` alias reads those entries and compiles
the listed `.slang` files into the specified `spv_dir`.

Example shader outline:

.. code-block:: c

   // swish.slang (conceptual)
   [numthreads(256, 1, 1)]
   void swish_f32_normal(uint3 tid : SV_DispatchThreadID) {
     uint idx = tid.x;
     if (idx >= numel) return;
     TensorDesc a = tensor_descs[0];
     TensorDesc out = tensor_descs[1];
     float x = load_f32(a, idx, data);
     float y = x / (1.0 + exp(-beta * x));
     store_f32(out, idx, y);
   }

The Vulkan backend uses shared helpers for dtype conversion and packed types.
If your op supports packed types, you will need to use helpers from
`packed_utils.slang` to decode and encode values.

Step 5: Register kernels
------------------------

OpenInfer uses registries to map `(op, dtypes, mode, device)` to a kernel
function or shader. After implementing the CPU and Vulkan kernels, add entries
to the appropriate registry files.

Key points:

- The registry key includes the op name and dtype tuple.
- If the op is in-place or accumulation-capable, register those variants too.
- Vulkan registry entries include the shader entrypoint name.

Example (actual pattern):

.. code-block:: rust

   pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
       build_op_entries_same_input(OpKind::Add, |mode| match mode {
           OpMode::Normal => Some(add_normal_dispatch),
           OpMode::Inplace => Some(add_inplace_dispatch),
           OpMode::Accumulate => Some(add_accumulate_dispatch),
       })
       .expect("failed to build add entries")
   });

The CPU and Vulkan registries both use this pattern. The per-op `registry.rs`
exports `ENTRIES`, and the top-level registry aggregates them.

Step 6: Update validation rules
-------------------------------

Validation ensures that a graph using the new op is safe to execute. You should
review:

- Attribute validation (types and ranges).
- Dtype compatibility checks.
- Output dtype inference rules.

If your op has non-trivial dtype rules (e.g. output dtype depends on input dtype
or an attribute), implement that logic in the validation layer.

Step 7: Add tests and baselines
-------------------------------

Tests should cover:

- CPU kernel correctness for each dtype.
- Vulkan kernel correctness (if supported).
- In-place and accumulation modes (if supported).
- Broadcast behavior (if supported).

OpenInfer uses baseline data for ops. Add a new test case to the ops baseline
generator if your op should be included in baseline validation.

Example test workflow:

.. code-block:: bash

   ./scripts/run_tests.sh --test-filter openinfer::ops_basic
   ./scripts/run_tests.sh --target=vulkan --features=vulkan --test-filter openinfer::ops_basic

Step 8: Document the op
-----------------------

Finally, update documentation:

- Add or update the op entry in `ops.json` (already done).
- Regenerate the ops catalog (`docs/sphinx/generate_ops_docs.py`).
- Add a short description and example to the relevant Sphinx guide if needed.

If the op is important or non-trivial, consider adding a short example snippet
to `guides/ops.rst` or a module guide. This makes the new op discoverable.

Common pitfalls
---------------

- **Mismatched names**: The DSL name must match the registry name and the
  schema name.
- **Missing dtype variants**: If a dtype is listed in `ops.json` but no kernel
  is registered, execution will fail at runtime.
- **Attribute type mismatch**: Attribute parsing preserves types. If the schema
  expects `f32` but you pass an int literal, validation will fail.
- **Packed dtype handling**: Packed types require explicit encoding/decoding in
  both CPU and Vulkan implementations.

Checklist
---------

Before you submit:

- `ops.json` updated and validated.
- CPU kernel implemented and registered.
- Vulkan shader implemented and registered (if supported).
- Validation rules updated for attributes and dtypes.
- Tests added or baselines updated.
- Ops catalog regenerated and docs updated.

This lifecycle may feel long, but it is the reason OpenInfer can keep a large
ops surface area consistent across CPU and Vulkan backends.

Deep dive: dtype and attribute rules
------------------------------------

Op schemas control dtype and attribute compatibility. This is where many
integration bugs happen, so it is worth understanding how the schema is used.

When the executor runs a `NodeKind::Op`, it resolves input tensors and infers
their dtypes. It then checks the schema to see if the dtype combination is
supported for the given device. If the combination is not supported, execution
fails early with a descriptive error.

Attributes are validated by type and sometimes by range. For example, if the
schema declares `beta: f32`, then the DSL must provide a float literal or a
float-valued variable. If you provide an integer literal, validation fails. This
strictness prevents silent bugs where the runtime would otherwise cast or clamp
values without your knowledge.

When you introduce a new op with attributes, implement validation rules in the
same place as other op validations. That keeps the system consistent and makes
errors easy to trace.

Deep dive: registry flow
------------------------

Registries are the bridge between op names and kernel implementations. The
registry key usually includes:

- Op kind (name).
- Device (CPU or Vulkan).
- Dtype (and sometimes input dtype tuple).
- Op mode (normal, in-place, accumulate).

The dispatcher builds the key from the graph node, then queries the registry.
If the registry does not contain an entry, it returns a descriptive error. This
means you must register every dtype variant you list in `ops.json`.

When you add a new op, verify the registry entries for each dtype and device.
Do not assume the registry will infer missing entries.

Example: incremental validation
--------------------------------

A practical workflow for adding a new op is:

1. Add the op to `ops.json`.
2. Implement the CPU kernel for a single dtype (e.g., `f32`).
3. Register only that dtype and run CPU tests.
4. Add additional dtypes.
5. Implement Vulkan for one dtype.
6. Expand Vulkan coverage and test.

This incremental approach keeps the debugging surface small. If you jump
directly to full dtype coverage across CPU and Vulkan, you will likely spend
more time diagnosing failures.

Documentation expectations
--------------------------

When adding a new op, update the documentation so the op is discoverable:

- The ops catalog will include it automatically after regeneration.
- Add a short example to `guides/ops.rst` or a module guide if the op is
  non-trivial or has unique attributes.
- If the op introduces a new dtype rule or attribute pattern, document it.

The goal is that a developer can find the op in Sphinx, see its signature, and
understand how to use it without reading the code.
