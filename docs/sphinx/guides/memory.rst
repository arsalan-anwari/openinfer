Memory Model
============

OpenInfer treats memory as an explicit part of the graph definition. Variables
are declared in memory sections and used by nodes.

Memory kinds
------------

- `dynamic`: provided at runtime and cleared each step.
- `volatile`: mutable during execution and reset each step.
- `constant`: immutable values loaded from the model package.
- `persistent`: mutable state that survives across steps.

The memory kind defines both lifetime and mutability. For example, constants are
loaded from the model and never written, while persistent values can be updated
across execution steps (useful for KV caches and recurrent state).

Cache tables
------------

Persistent variables can be treated as cache tables. Cache operations include:

- `cache.read`
- `cache.write`
- `cache.increment` / `cache.decrement`
- `cache.reset`

Use caches for attention KV buffers, rolling windows, or other persistent state.

Cache tables are indexed by symbolic variables (e.g., `layer`, `time`) that
appear in the DSL. This makes indexing explicit and easy to analyze.

Interacting with memory
-----------------------

- `dynamic` variables can only be mutated via executor inputs.
- `volatile`, `constant`, and `persistent` can be fetched during execution, but
  only `persistent` and `volatile` are mutable within the graph.

The executor API is the bridge between host data and graph variables. Use
`insert_executor!` to populate `dynamic` inputs and `fetch_executor!` to read
results or state.

Variable attributes
-------------------

Attributes on variable definitions describe metadata and linkage:

.. code-block:: rust

   constant {
     alpha: f32 @ref("alpha");
     beta:  f32 @ref("beta");
   }

Prefix tables
-------------

Prefix tables map indexed DSL names to model tensor families:

.. code-block:: rust

   volatile {
     W(l): f32[D, D] @pattern("W.{l}");
   }

This enables a single DSL declaration to bind to `W.0`, `W.1`, ... from the
model package.

Persistent tables
-----------------

Persistent caches can be tables with fixed or dynamic dimensions:

.. code-block:: rust

   persistent {
     K(l, t): f16[H, Dh] @table;
     V(l, t): f16[H, Dh] @table;
   }

Auto-dim caches
---------------

Auto-dim tables grow as indices increase, useful for KV cache growth:

.. code-block:: rust

   persistent {
     rows: i32 @init(0);
     cols: i32 @init(0);
     M(r, c): f16[D, H] @auto_dim(r, c);
   }

   block entry {
     cache.increment 3 rows;
     cache.increment 5 cols;
     cache.read M[rows, cols] >> out;
     return;
   }

This pattern is common for streaming workloads where the time dimension grows
as new tokens arrive.

Memory reuse
------------

Volatile assignments are ephemeral and may be reused or aliased by the runtime.
This enables efficient memory usage without changing graph semantics.

When debugging numerical issues, consider disabling reuse in the executor
configuration to simplify tracing and buffer lifetimes.

Reuse trade-offs
----------------

Memory reuse reduces peak allocation but can affect per-op throughput. Packed
types and accumulate modes may introduce extra work. This is an intentional
trade-off to keep large models runnable on constrained systems.
