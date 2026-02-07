Project Overview
================

OpenInfer is an edge-focused ML transpilation framework. It lets developers
express inference pipelines in a Rust-embedded DSL, validate them in a
high-level simulator, and then synthesize fully static, device-specific source
code for deployment on constrained hardware.

The simulator runs purely on the host to verify graph correctness, scheduling,
memory layouts, and DSL transformations. A planned synthesizer lowers the same
graph into concrete C/CUDA/Vulkan/VHDL-style source that targets a thin hardware
abstraction layer and native device APIs.

Core flow
---------

1. Author graphs with the DSL (`openinfer-dsl`).
2. Validate graphs and memory layouts in the simulator (`openinfer-simulator`).
3. Package models as `.oinf` files using Python tooling (`openinfer-oinf`).
4. Lower graphs to static, device-specific code (`openinfer-synth`).

User method (packages)
----------------------

Install from package managers (recommended for users):

Rust crates:

.. code-block:: bash

   cargo add openinfer-simulator
   cargo add openinfer-dsl
   cargo add openinfer-synth

Python tooling:

.. code-block:: bash

   pip install openinfer-oinf

Modules
-------

- `openinfer-simulator`: host-side runtime + validation.
- `openinfer-dsl`: Rust-embedded graph DSL.
- `openinfer-oinf`: Python tooling for `.oinf` model packages.
- `openinfer-synth`: codegen pipeline for edge targets.

Where to go next
----------------

- Use the module menus in the sidebar for module-specific guides and APIs.

