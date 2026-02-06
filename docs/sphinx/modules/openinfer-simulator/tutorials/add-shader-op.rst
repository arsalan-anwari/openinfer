Add a Shader Op
===============

Walkthrough for adding a Vulkan shader-backed op.

Outline
-------

- Add shader sources
- Register SPIR-V map entry
- Validate with tests

1. Add shader source
--------------------

Create a Slang shader under `openinfer-simulator/src/ops/vulkan/<category>/<op>/`.

.. code-block:: c

   [shader("compute")]
   void <op_name>_<dtype>_<mode>() {
       // Kernel logic.
   }

2. Register SPIR-V
------------------

Update the SPIR-V mapping to include the shader entry point.

3. Kernel wiring
----------------

Add the Vulkan kernel registration for the op kind and mode.

4. Tests
--------

Add a test that compares CPU and Vulkan outputs where applicable.
