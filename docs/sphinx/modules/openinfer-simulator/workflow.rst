Workflow
========

How to build and test the simulator crate (`openinfer` package in
`openinfer-simulator/`).

Topics
------

- Building CPU-only vs Vulkan
- Running core tests

Build
-----

.. code-block:: bash

   cargo build -p openinfer

Vulkan:

.. code-block:: bash

   cargo build -p openinfer --features vulkan

Tests
-----

.. code-block:: bash

   cargo test -p openinfer

Run an example
--------------

.. code-block:: bash

   cargo run --example mlp_regression


