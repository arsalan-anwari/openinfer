Add a New Op
============

Walkthrough for adding a new op kind to the runtime.

Outline
-------

- Add op schema entry
- Add CPU kernel implementation
- Add tests and baselines

1. Add schema in `ops.json`
---------------------------

Define name, arity, attrs, and dtype support. Keep the name in snake_case.

.. code-block:: json

   {
     "name": "add",
     "category": "arithmetic",
     "inputs": { "arity": "fixed", "count": 2 },
     "outputs": { "arity": "fixed", "count": 1 },
     "attrs": [],
     "broadcast": "allow",
     "inplace": "allow",
     "accumulate": "allow",
     "type_rule": { "kind": "same_as_input", "index": 0 }
   }

2. Implement CPU kernel
-----------------------

Create a kernel under `openinfer/src/ops/cpu/<category>/<op>/kernel.rs` and wire
it into the registry for the op category.

.. code-block:: rust

   pub fn add_normal_dispatch(
       attrs: &OpAttrs,
       inputs: &[TensorValue],
       output: Option<&mut TensorValue>,
   ) -> Result<()> { /* ... */ }

3. Optional Vulkan shader
-------------------------

Add shader sources under `openinfer/src/ops/vulkan/...` and update SPIR-V
mapping so the kernel is discoverable.

4. Tests and baselines
----------------------

Add tests in `tests/openinfer/ops` and generate baseline data if the op is part
of baseline suites.
