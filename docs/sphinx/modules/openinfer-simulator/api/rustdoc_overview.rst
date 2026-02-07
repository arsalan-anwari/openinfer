Rustdoc Overview
================

OpenInfer uses native Rustdoc for API-level documentation. The Sphinx portal
links to those HTML pages instead of re-rendering them.

How Rustdoc is built
--------------------

The build script runs:

.. code-block:: bash

   cargo doc --manifest-path openinfer-simulator/Cargo.toml --no-deps
   cargo doc --manifest-path openinfer-dsl/Cargo.toml --no-deps
   cargo doc --manifest-path openinfer-simulator/generator/Cargo.toml --no-deps

Rustdoc output is copied into:

``docs/sphinx/out/api/rustdoc``

Why not sphinxcontrib-rust?
---------------------------

`sphinxcontrib-rust` can generate RST from Rust sources, but native Rustdoc is
the authoritative API documentation and stays closer to Rust conventions.

Links
-----

- :doc:`rustdoc` for per-crate links.

