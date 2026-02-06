Internals
=========

Key subsystems and data flows inside the core runtime.

Topics
------

- Graph blocks and node execution
- Tensor storage and dtype handling
- Runtime validation and tracing

Graph execution
---------------

- Nodes are executed in order within a block.
- `branch` and `loop` nodes push new frames onto the execution stack.
- `yield` and `await` integrate with the async scheduler.

Validation pipeline
-------------------

The runtime validates:

- Variable names, shapes, and dtypes against the model.
- Op attributes and dtype rules.
- Control flow constraints.

Op dispatch
-----------

`exec_op` selects a kernel based on:

- Op kind
- Input dtypes
- Output dtype (type rule or `acc` attribute)
- Broadcast/in-place/accumulate flags

Lazy loading
------------

`ModelLoader::open` memory-maps `.oinf` files and records offsets. Tensor data
is loaded on demand when an op consumes a variable.

Tracing
-------

Trace events store node index, UUID, block name, op name, and timing.


