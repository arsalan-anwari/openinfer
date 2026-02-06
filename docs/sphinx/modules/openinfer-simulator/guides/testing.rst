Testing
=======

OpenInfer uses both Rust and Python tests. This guide summarizes how to run
and extend them.

Rust tests
----------

Run the full Rust test suite:

.. code-block:: bash

   cargo test --manifest-path openinfer-simulator/Cargo.toml

Many tests compare outputs to baseline JSON/OINF files. Keep baselines up to
date when modifying ops or dtype behavior.

Baseline data
-------------

Baseline data is stored under:

``openinfer-simulator/tests/ops/baseline/data``

If you update ops or serialization formats, regenerate baselines and keep
changes minimal and documented.

NumPy baselines
~~~~~~~~~~~~~~~

Most op tests use a NumPy reference to validate correctness. The baseline
generator (`openinfer-simulator/tests/ops/baseline/gen_ops_baseline.py`) builds inputs
and expected outputs with NumPy, then writes them into `.oinf` baselines. Rust
tests load those baselines and compare device results against the NumPy outputs.

Comparison behavior:

- Float outputs are checked with dtype-specific tolerances (looser for f16/f8,
  stricter for f32/f64).
- Vulkan targets allow a slightly larger tolerance to account for device
  variance.
- Integer, boolean, and packed types must match exactly.

This keeps CPU, Vulkan, and future targets aligned with the same NumPy ground
truth.

Python verification
-------------------

The Python suite validates `.oinf` encoding/decoding:

.. code-block:: bash

   python openinfer-oinf/tests/run_oinf_tests.py

Common failure cases include dtype mismatches, packed type errors, and invalid
header offsets.

Test locations
--------------

- `openinfer-simulator/tests`: Rust runtime tests.
- `openinfer-dsl/tests`: DSL parsing tests.
- `openinfer-oinf/tests`: Python encoder/verify tests.
