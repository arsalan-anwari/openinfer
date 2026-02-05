Examples
========

The `examples/openinfer` directory contains runnable programs that demonstrate
common patterns.

Common examples
---------------

- `mlp_regression`: basic MLP graph and execution.
- `linear_attention`: attention kernel composition.
- `quantized_linear`: packed dtype usage.

Running examples
----------------

.. code-block:: bash

   cargo run --example mlp_regression

Python `.oinf` examples
-----------------------

The `examples/openinfer-oinf` directory mirrors the Rust examples and shows how
to generate `.oinf` files using Python tooling.
