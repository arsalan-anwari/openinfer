Workflow
========

How to build and test the DSL crate.

Topics
------

- Running parser tests
- Updating syntax and fixtures

Build and tests
---------------

.. code-block:: bash

   cargo test -p openinfer-dsl

Parser changes
--------------

When adding syntax:

1. Update keywords and parsers in `openinfer-dsl`.
2. Update validation logic.
3. Add or update tests under `openinfer-dsl/tests`.

