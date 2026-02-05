openinfer-oinf (Python API)
==========================

The `openinfer-oinf` package provides Python tooling for authoring, validating,
and inspecting `.oinf` model files. It is intentionally small and script-driven:
the core workflows are expected to be run from the command line during
development, while the API is thin enough to embed into build steps or custom
model export pipelines.

This page describes the public API surface and the intended usage patterns in
more detail than the CLI help. If you are new to the project, start with the
`Getting Started` and `OINF Format` guides to understand the binary layout and
the runtime expectations, then come back here for the concrete Python entry
points.

Overview of the Python tools
----------------------------

The Python tooling serves three core use cases:

1. **Author a model** by mapping Python data structures to the `.oinf` schema.
2. **Validate a model** by checking alignment, table integrity, and sizes.
3. **Inspect a model** by printing a human-readable summary and statistics.

The main scripts live under `openinfer-oinf/` and are used by example generators
in `examples/openinfer-oinf/`. The examples demonstrate a practical end-to-end
workflow: create an in-memory model, write it to disk, and verify the binary.

Key entry points
----------------

While the project uses scripts, the API is designed to be imported by those
scripts. The most commonly used functions and types include:

- `SizeVar`: a wrapper that marks a field as a size variable.
- `TensorSpec`: a wrapper that marks a field as a tensor payload.
- `write_oinf`: a high-level function that serializes a dataclass instance into
  a `.oinf` file.
- `verify_oinf`: a validation function that checks a `.oinf` file and prints
  a summary.

The examples in `examples/openinfer-oinf/` show these in action and are the best
reference for how the tooling is intended to be used in practice.

Creating a model from a dataclass
---------------------------------

The recommended authoring pattern is to define a dataclass that represents your
model. Fields can be plain values (metadata), `SizeVar` instances (sizevars), or
`TensorSpec` instances (tensors). This keeps the model declarative and easy to
inspect.

High-level workflow:

1. Define the dataclass and its fields.
2. Create an instance with concrete values and NumPy arrays.
3. Call `write_oinf(...)` to serialize to disk.
4. Call `verify_oinf(...)` to validate and print a summary.

Example:

.. code-block:: python

   from dataclasses import dataclass
   import numpy as np

   from dataclass_to_oinf import SizeVar, TensorSpec, write_oinf

   @dataclass
   class MlpModel:
       B: SizeVar
       D: SizeVar
       H: SizeVar
       O: SizeVar
       w1: TensorSpec
       b1: TensorSpec
       w2: TensorSpec
       b2: TensorSpec

   def build() -> MlpModel:
       rng = np.random.default_rng(1)
       B, D, H, O = 4, 16, 32, 8
       return MlpModel(
           B=SizeVar(B),
           D=SizeVar(D),
           H=SizeVar(H),
           O=SizeVar(O),
           w1=TensorSpec(rng.normal(size=(D, H)).astype(np.float32)),
           b1=TensorSpec(rng.normal(size=(H,)).astype(np.float32)),
           w2=TensorSpec(rng.normal(size=(H, O)).astype(np.float32)),
           b2=TensorSpec(rng.normal(size=(O,)).astype(np.float32)),
       )

   write_oinf(build(), "mlp.oinf")

This pattern keeps sizes and tensor data co-located in one object, which makes
the binary layout deterministic and easy to validate. If you need to encode
additional metadata, add extra fields to the dataclass using plain Python types.

Metadata and scalar handling
----------------------------

The serializer uses field types to choose the correct table:

- Size vars go into the **SizeVarsTable**.
- Scalars and small values go into the **MetadataTable**.
- Tensor buffers go into the **TensorIndexTable** + **DataBlobs**.

If you need a scalar tensor (a tensor with `ndim = 0`), wrap the scalar value
with `TensorSpec` rather than leaving it as metadata. This ensures it is treated
as a tensor at runtime, not as metadata.

Example: scalar tensor vs metadata

.. code-block:: python

   @dataclass
   class Example:
       scale_meta: float
       scale_tensor: TensorSpec

   model = Example(
       scale_meta=0.1,
       scale_tensor=TensorSpec(np.array(0.1, dtype=np.float32)),
   )

The runtime will only load `scale_tensor` as a tensor buffer. The metadata value
can still be read for configuration but is not treated as a runtime tensor.

Validation and inspection
-------------------------

The verifier performs a structural validation before printing a summary:

- Verifies header magic/version and table offsets.
- Checks alignment to 8 bytes for tables and blobs.
- Validates strings against the ASCII character set.
- Ensures tensor data sizes match `numel * sizeof(dtype)`.

Example CLI workflow:

.. code-block:: bash

   python openinfer-oinf/verify_oinf.py res/models/mlp_regression.oinf

The output includes:

- Size vars as `NAME := VALUE`.
- Metadata values with explicit types.
- Tensors with a short preview and statistics (min/max/mean).

If the verifier fails, use the printed offsets and tensor names to locate the
bad entry. This is particularly useful when generating models programmatically.

Embedding in custom pipelines
-----------------------------

The Python tools are intentionally dependency-light so they can be used in CI
or as part of a build pipeline. Typical integration points include:

- Exporting a model from training code to `.oinf`.
- Verifying the model as part of a CI step.
- Producing a short summary for release artifacts.

If you need specialized behavior (custom metadata serialization, alternative
layout rules, or additional validation), start by wrapping or extending the
existing script functions rather than rewriting from scratch. The OINF format is
strict about alignment and offsets, so reuse reduces the chance of subtle bugs.

Relationship to the runtime
---------------------------

The runtime expects:

- Names in the `.oinf` file to match DSL variable names.
- Dims to be resolvable via size vars.
- Tensor dtypes to be compatible with op schemas and kernels.

The Python tools do not validate graph compatibility. That is the runtime’s
responsibility, so make sure to run the simulator or at least the verifier for
structural correctness.

Where to go next
----------------

- `guides/oinf-format` for the binary layout details.
- `guides/serialization` for how graphs are serialized to JSON.
- `examples/openinfer-oinf` for end-to-end model generation examples.
