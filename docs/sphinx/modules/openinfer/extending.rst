Extending
=========

How to add functionality to the core runtime.

Topics
------

- Adding a new op kind
- Adding a new runtime backend

Add a new op
------------

1. Add schema in `ops.json` with arity, attrs, and dtype support.
2. Implement CPU kernel under `openinfer/src/ops/cpu/...`.
3. Register Vulkan shader (optional) and SPIR-V entry.
4. Add tests and baseline data.

Add a new backend
-----------------

Backends are selected by `Device` and the kernel registry. Implement a registry
for the new backend and ensure kernels are warmed for device selection.
