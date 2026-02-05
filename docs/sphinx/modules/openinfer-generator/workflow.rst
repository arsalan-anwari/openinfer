Workflow
========

The generator is used in offline workflows. A typical flow:

1. Update the canonical schema source (for ops this is `ops.json`).
2. Run the generator from a script.
3. Commit the generated output or re-run as part of CI.

Generator inputs
----------------

Inputs are treated as structured data with explicit typing and validation. This
keeps the output stable and makes failures actionable.

Deterministic output
--------------------

Outputs are generated with deterministic ordering to avoid noisy diffs. Where
possible, ordering uses:

- stable keys (op name, category)
- explicit sort keys defined in the generator

Example: ops catalog generation
-------------------------------

The ops catalog in Sphinx is generated from `ops.json`:

.. code-block:: bash

   python docs/sphinx/generate_ops_docs.py

This script:

1. Loads and validates `ops.json`.
2. Groups ops by category and capability.
3. Writes a category index and per-op pages.
