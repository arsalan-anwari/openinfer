Testing
=======

OpenInfer uses both Rust and Python tests. This guide summarizes how to run
and extend them.

Rust tests
----------

Run the full Rust test suite:

.. code-block:: bash

   cargo test

Many tests compare outputs to baseline JSON/OINF files. Keep baselines up to
date when modifying ops or dtype behavior.

Baseline data
-------------

Baseline data is stored under:

``tests/openinfer/ops/baseline/data``

If you update ops or serialization formats, regenerate baselines and keep
changes minimal and documented.

Python verification
-------------------

The Python suite validates `.oinf` encoding/decoding:

.. code-block:: bash

   python tests/openinfer-oinf/run_oinf_tests.py

Common failure cases include dtype mismatches, packed type errors, and invalid
header offsets.

Test locations
--------------

- `tests/openinfer`: Rust runtime tests.
- `tests/openinfer-dsl`: DSL parsing tests.
- `tests/openinfer-oinf`: Python encoder/verify tests.
