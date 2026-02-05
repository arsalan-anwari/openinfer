Workflow
========

How to build and test the `openinfer` crate.

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
