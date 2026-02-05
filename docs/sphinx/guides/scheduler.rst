Control Flow and Scheduling
===========================

This chapter describes how OpenInfer schedules and executes graph nodes. It
expands on the shorter `Control Flow` guide and focuses on the runtime’s actual
execution model: how blocks are entered, how nodes are ordered, how loops and
branches are handled, and how yield/await affects scheduling. The goal is to
make it clear *when* a node runs and *why* it runs in that order.

OpenInfer intentionally avoids a complex hidden scheduler. Execution is driven
directly by the graph structure, which is why the DSL exposes control-flow
features explicitly. This explicit model is the reason OpenInfer is easier to
debug than many runtimes: if you want to change execution order, you update the
graph.

Execution phases
----------------

At a high level, execution proceeds in three phases:

1. **Validation**: the simulator checks the graph against the model and
   configuration. It verifies dtypes, sizevars, attributes, and memory rules.
2. **Preparation**: the executor initializes variable storage (including lazy
   loading of constants) and builds internal state for blocks and tables.
3. **Execution**: the executor steps through blocks, dispatching ops and control
   flow nodes in graph order.

The key is that the graph itself encodes control flow. There is no hidden
optimizer. The executor does not reorder nodes across barriers or deps. It does
not invent branches. It simply follows the `NodeKind` list inside each block,
and uses explicit control-flow nodes to determine which block to run next.

Block scheduling model
----------------------

Each block is an ordered list of nodes. The executor maintains a current block
and a current node index. When the block finishes, control-flow nodes determine
the next block or whether execution halts.

The simplest case is a single block:

.. code-block:: rust

   block entry {
     op add(x, y) >> z;
     op relu(z) >> z;
     return;
   }

Here the executor runs nodes in order and stops at `return`.

Branches introduce explicit jumps:

.. code-block:: rust

   block entry {
     branch cond ok bad;
   }

   block ok { ... }
   block bad { ... }

The `branch` node evaluates `cond`, then sets the current block to either `ok`
or `bad`. That is the entirety of the scheduling logic. There is no hidden
heuristic or probabilistic scheduling; the branch node defines the decision.

Loops are implemented as control-flow nodes that repeat a block or sequence of
nodes. The loop construct expands into a `Loop` node that contains the loop
bounds and the inner block. The executor keeps a loop counter and re-enters the
loop body until the end condition is met.

Yield/await scheduling
----------------------

Yield and await are the primary tools for coordinating execution across blocks
in asynchronous or streaming scenarios. They are explicit nodes:

- `yield` publishes values and signals that the current block can pause.
- `await` blocks until the value is available from another block.

The runtime uses these nodes to coordinate execution without adding implicit
locks. This is critical for deterministic scheduling: every await has a matching
yield, and the dependency is explicit in the graph.

Example:

.. code-block:: rust

   block producer {
     op compute(x) >> out;
     yield out;
   }

   block consumer {
     await out;
     op consume(out) >> y;
     return;
   }

The scheduler will not run `consumer` beyond the `await` until `producer` has
yielded `out`. The key point is that the graph encodes this dependency; there is
no separate dependency graph computed by the runtime.

Ordering with barrier and dep
-----------------------------

The runtime maintains order within a block, but you can also enforce ordering
across nodes that do not share data edges. This is the role of `barrier` and
`dep`.

**barrier** inserts a hard execution boundary. No node after the barrier should
be reordered before it. It is used to express side-effect ordering.

**dep** creates a dependency edge between two variables without copying data.
This is useful when you want to enforce that an op runs after another even if
there is no data dependency. This is common for cache operations or resource
management nodes.

Example:

.. code-block:: rust

   op update_cache(x) >> tmp;
   dep after(tmp) before(y);
   op compute(y, tmp) >> z;

The `dep` node tells the scheduler that `compute` must wait until `update_cache`
finishes, even if `y` is not produced by the previous op.

How nodes are executed
----------------------

The executor processes nodes by type. The key categories are:

- **Assignment**: allocate a temporary tensor.
- **Op**: dispatch a kernel based on op type, dtypes, and device.
- **Cache**: perform table read/write or index updates.
- **Control Flow**: branch, loop, yield, await, return.

For op nodes, the executor:

1. Resolves input tensors (loading constants lazily).
2. Resolves output tensor (allocate or reuse).
3. Looks up the kernel in the registry.
4. Executes the kernel on CPU or Vulkan.

For cache nodes, the executor:

1. Resolves the index values.
2. Locates the table entry (and grows if auto-dim).
3. Reads or writes the tensor.

For control-flow nodes, the executor updates its internal block state and moves
to the next block or returns.

This is all deterministic. If two graphs have the same nodes and inputs, they
will run in the same order.

Trace-driven understanding
--------------------------

OpenInfer provides trace logs that include node indices, block names, op names,
and timing. This allows you to see the execution order directly. If you are
unsure about scheduling behavior, the trace is the authoritative source.

Example command:

.. code-block:: bash

   OPENINFER_TRACE=full cargo run --example streaming_pipeline

Once you have the trace, inspect the sequence of nodes and compare it with the
DSL. The indices should match the insertion order in the graph. If they do not,
that is a bug in either the graph construction or the scheduler.

Practical guidance
------------------

When authoring graphs, keep these rules in mind:

- Use explicit blocks and branches for control flow.
- Use `barrier` and `dep` when ordering matters without data edges.
- Use `yield`/`await` for streaming and async patterns.
- Avoid relying on hidden behavior; if it matters, encode it in the DSL.

OpenInfer's scheduling is intentionally simple and explicit. The benefit is that
you can reason about execution from the graph alone. The cost is that you must
be precise when authoring the DSL. This guide exists to help you do that with
confidence.

Advanced scheduling notes
-------------------------

While the execution model is explicit, there are still subtle interactions
between blocks, loops, and yield/await. These are worth understanding if you
build complex streaming graphs.

**Non-entry blocks**

Only the `entry` block is executed automatically. All other blocks must be
entered via `branch`, `loop`, or scheduler logic driven by `yield`/`await`.
If a block is never referenced, it is effectively dead code. This is useful for
debugging: you can leave experimental blocks in the graph and only branch into
them when needed.

**Yield/await ownership**

The runtime expects that a `yield` produces a value that an `await` will consume.
If an await is encountered without a matching yield, execution will block or
error. The exact behavior depends on the scheduler implementation, but the key
idea is that yield/await define explicit synchronization points.

**Loop boundaries and state**

Loop constructs are explicit and do not imply any caching or variable resets.
If you need to reset a variable each loop iteration, you must do it explicitly.
Likewise, if you need persistent state across iterations, you must use
`persistent` variables or caches.

These constraints are intentionally strict because they allow you to reason
about execution without hidden state. The cost is verbosity, but the benefit is
determinism and traceability.

Example: structured pipeline
----------------------------

The following example shows a pipeline with a producer and consumer block that
coordinate using yield/await. It also illustrates how to structure blocks to
make scheduling intent clear:

.. code-block:: rust

   block entry {
     branch ready producer idle;
   }

   block producer {
     op compute(x) >> out;
     yield out;
     branch more producer done;
   }

   block consumer {
     await out;
     op consume(out) >> y;
     return;
   }

   block idle {
     return;
   }

   block done {
     return;
   }

This example is intentionally verbose. The explicit branches make it obvious
which blocks can run and when they exit. When you read the trace, you can follow
the block transitions directly.
