Operations
==========

Operations are defined in `ops.json` and loaded at runtime.

Catalog
-------

See the generated ops catalog for per-op details:

- :doc:`../ops/index`

Adding new ops
--------------

For the complete lifecycle (schema, CPU, Vulkan, registry, tests), see:

- :doc:`adding-ops`

Regeneration
------------

The ops catalog is generated from `ops.json` by:

.. code-block:: bash

   python docs/sphinx/generate_ops_docs.py

This is run automatically by `./scripts/build_docs.sh` before Sphinx builds.

What ops define
---------------

- Name and arity.
- Attributes and attribute types.
- Broadcast/in-place/accumulate support.
- DType coverage for CPU and Vulkan.
- Output dtype inference rules.

Current op set
--------------

The current op list includes arithmetic, reduction, bitwise, comparison,
filtering, and casting ops. See `ops.json` for the authoritative list.

Highlights:

- Arithmetic: `add`, `mul`, `sub`, `div`, `fma`, `neg`, `recip`
- Reductions: `sum_axis`, `mean_axis`, `prod_axis`, `max_axis`, `min_axis`
- Comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Casting: `cast`

Adding ops
----------

1. Add schema in `ops.json`.
2. Implement CPU kernel(s).
3. Add Vulkan shader (optional).
4. Add tests and baseline data.

Output dtype rules
------------------

Output dtypes are derived from input dtypes or attributes:

- `same_as_input`: output dtype matches an input.
- `fixed`: output dtype is constant.
- `acc_from_attr`: output dtype is defined by an `acc` attribute.
