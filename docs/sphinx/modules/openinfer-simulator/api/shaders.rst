Shaders
=======

OpenInfer uses Slang shaders compiled to SPIR-V for Vulkan compute ops. Shaders
are organized by op category and support multiple data types and modes.

Shader organization
-------------------

Shaders live under:

``openinfer-simulator/src/ops/vulkan/<category>/<op>/shaders/``

Common files:

- `normal.slang`: standard modes
- `packed.slang`: packed dtype modes
- `accumulate.slang`: accumulation modes
- `common.slang`: per-op push constants and helpers

Entrypoint naming
-----------------

Entrypoints follow: `<op>_<dtype>_<mode>`

- `add_f32_normal`
- `add_f32_inplace`
- `add_i4_packed`
- `sum_axis_f32_accumulate`

Example:

.. code-block:: c

   [numthreads(256, 1, 1)]
   [shader("compute")]
   void add_f32_normal(uint3 tid : SV_DispatchThreadID) {
       // Kernel logic.
   }

Descriptor bindings
-------------------

All compute shaders use a standard layout:

- Binding 0: `StructuredBuffer<TensorDesc>`
- Binding 1: `ByteAddressBuffer` inputs
- Binding 2: `RWByteAddressBuffer` outputs

Shared utilities
----------------

Common helpers live in:

- `openinfer-simulator/src/ops/vulkan/shaders/common.slang`
- `openinfer-simulator/src/ops/vulkan/shaders/packed_utils.slang`
- `openinfer-simulator/src/ops/vulkan/shaders/float_utils.slang`
- `openinfer-simulator/src/ops/vulkan/shaders/reduce_utils.slang`

Build workflow
--------------

Compile shaders to SPIR-V and embed them:

.. code-block:: bash

   cargo build-spv

See also
--------

- :doc:`../tutorials/add-shader-op`
- :doc:`../guides/ops-and-extensions`
