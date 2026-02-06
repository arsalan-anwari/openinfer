openinfer-oinf (Python API)
===========================

The `openinfer-oinf` package provides Python tooling for authoring, validating,
and inspecting `.oinf` model files. The API is intentionally light-weight so it
can be embedded in build steps and export pipelines for edge deployments.

Quick entry points
------------------

- `dataclass_to_oinf.write_oinf`: serialize a dataclass instance into `.oinf`.
- `dataclass_to_oinf.SizeVar`: mark size variables.
- `dataclass_to_oinf.TensorSpec`: mark tensor payloads.
- `oinf_verify.parse_file`: validate and print `.oinf` contents.

Autodoc modules
---------------

.. automodule:: dataclass_to_oinf
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: oinf_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: oinf_verify
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: oinf_common
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: oinf_types
   :members:
   :undoc-members:
   :show-inheritance:
