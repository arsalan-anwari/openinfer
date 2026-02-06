Project Overview
================

OpenInfer is an edge-focused ML transpilation framework. It lets you express
inference pipelines in a Rust-embedded DSL, validate them in a host-side
simulator, and then synthesize fully static, device-specific source code for
deployment on constrained hardware.

Core flow
---------

1. Author graphs with the DSL (`openinfer-dsl`).
2. Validate graphs and memory layouts in the simulator (`openinfer-simulator`).
3. Package models as `.oinf` files using Python tooling (`openinfer-oinf`).
4. Lower graphs to static, device-specific code (`openinfer-synth`).

Modules
-------

- `openinfer-simulator`: host-side runtime + validation.
- `openinfer-dsl`: Rust-embedded graph DSL.
- `openinfer-oinf`: Python tooling for `.oinf` model packages.
- `openinfer-synth`: codegen pipeline for edge targets.

Where to go next
----------------

- Use the module menus in the sidebar for module-specific guides and APIs.

