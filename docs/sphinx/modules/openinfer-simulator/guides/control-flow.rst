Control Flow
============

OpenInfer graphs are control-flow graphs, not just linear op lists.

Blocks
------

Execution is defined in `block` sections. The `entry` block is the start point.
Blocks contain ordered nodes, and terminate with `return` or `yield`.

Blocks are the unit of control-flow. Each block has a name and a node list. A
block can jump to another block via `branch`, or repeat with `loop`. This makes
control flow explicit and makes tracing deterministic.

Assignments and ops
-------------------

- `assign` declares a temporary tensor or scalar.
- `op` executes a computation and writes to the output after `>>`.

.. code-block:: rust

   assign h: f32[B, D];
   op matmul(x, w) >> h;

Assignments can be used to create temporary buffers for intermediate values,
which keeps later nodes simple and avoids hidden allocations.

Branches
--------

Use `branch` to jump to another block:

.. code-block:: rust

   branch cond ok bad;

The target blocks must exist and are resolved by name. This structure makes
branches explicit in the graph and avoids implicit control flow.

Loops
-----

Loops are explicit:

.. code-block:: rust

   loop layers (l in 0..num_layers) { ... }

Loops can be unrolled or analyzed by tooling in the future because the DSL keeps
iteration bounds explicit and symbolically named.

Barriers and deps
-----------------

- `barrier;` prevents reordering across a boundary.
- `dep after(x) before(y);` enforces ordering without data edges.

.. code-block:: rust

   op matmul(x, w) >> h;
   barrier;
   op relu(h) >> h;

Use barriers when two ops touch shared state (like caches) but do not share
explicit tensor edges.

Yield/await
-----------

`yield` and `await` allow blocks to coordinate access to variables in async or
streaming execution modes.

.. code-block:: rust

   block entry {
     op matmul(x, w) >> h;
     yield h; // cannot use h again in this block
     op abs(x) >> x;
     await h; // can use h again in this block
     op add(x, h) >> h;
     return;
   }
   block consumer {
     await h;
     op relu(h, alpha=0.0, clamp_max=6.0) >> h;
     yield h;
   }

In streaming pipelines, `yield` can publish a tensor to a downstream consumer,
while `await` blocks until the value is ready.
