Caching System (Tables, Slices, Indices)
========================================

OpenInfer treats caches as first-class graph state. The caching system is not a
hidden optimization; it is a language feature. This guide explains cache tables,
slice indices, and the rules used by the runtime to update and access persistent
state. It is long by design because cache behavior is a frequent source of bugs
and performance surprises in stateful models, especially when you move from
batch to streaming execution.

The most important idea is that caching is *explicit*. If you read from a cache,
you will see a `cache.read` node. If you grow a cache, you will see
`cache.increment`. This allows you to reason about memory growth, index
progression, and ordering without guessing what the runtime does behind your
back.

Why caches are explicit
-----------------------

Many inference runtimes hide cache behavior inside the kernel or operator. That
approach can make a graph look simpler, but it also makes the execution harder
to debug and reason about. OpenInfer takes a different approach: cache behavior
is declared in the graph and enforced by validation. This gives you precise
control over the lifecycle of persistent state.

Caches matter when:

- You build streaming models with incremental tokens.
- You maintain recurrent state across steps.
- You reuse intermediate tensors across blocks.
- You need explicit control over memory growth.

In these cases, implicit caching quickly becomes ambiguous. Is a buffer reused?
When is it cleared? What is the index? OpenInfer's cache tables answer those
questions explicitly.

Cache tables and indexing
-------------------------

A cache table is a persistent variable with one or more *table indices*. In the
DSL, you declare it with index variables and mark it with `@table`:

.. code-block:: rust

   persistent {
     K(l, t): f16[H, Dh] @table;
     V(l, t): f16[H, Dh] @table;
   }

Here, `l` and `t` are *symbolic table indices*, not normal tensor dimensions.
The tensor itself still has dimensions `[H, Dh]`, but the table is indexed by
`(l, t)` and resolves to a specific tensor instance at runtime. Think of it as a
map keyed by `(l, t)` that returns a tensor buffer with a fixed shape.

This design decouples the *tensor shape* from the *table index*. The shape is
known and validated, while the index evolves over time. This is crucial for
streaming workloads where the time index grows per token but the tensor shape
stays constant.

Accessing a cache table uses a bracket syntax:

.. code-block:: rust

   cache.read K[layer, time] >> k_out;
   cache.write v_in >> V[layer, time];

The index values (`layer`, `time`) are variables in the graph, typically stored
as persistent or volatile scalars. You control them with cache increment and
decrement nodes.

Slice indices and auto-dim caches
---------------------------------

Many models need caches that grow dynamically. For example, a KV cache grows
with each new token. OpenInfer supports this via `@auto_dim` tables, which
automatically resize as indices increase.

The typical pattern is:

1. Declare scalar index variables to track the current size.
2. Declare an auto-dim table that uses those index variables.
3. Increment the index variables before reading or writing.

Example:

.. code-block:: rust

   persistent {
     rows: i32 @init(0);
     cols: i32 @init(0);
     M(r, c): f16[D, H] @auto_dim(r, c);
   }

   block entry {
     cache.increment 1 rows;
     cache.increment 1 cols;
     cache.write x >> M[rows, cols];
     return;
   }

The `@auto_dim` annotation tells the runtime that the table size is not fixed.
When you increment `rows` or `cols`, the runtime ensures the table can store
that index. If the index grows beyond the current allocation, the cache grows.
This behavior is deterministic and explicit, which makes it easy to reason about
memory usage.

Slice indices are a common pattern in streaming models. For example, you might
use `t` to represent the current time step, and you might want to read the last
`W` entries. OpenInfer expresses this by indexing with `t - W` or by reading
multiple indices in a loop. The important point is that *you* express the slice
logic in the DSL; the runtime does not assume any slicing semantics.

Cache operations and semantics
------------------------------

Cache operations are explicit nodes. The core operations are:

- `cache.read`
- `cache.write`
- `cache.increment`
- `cache.decrement`
- `cache.reset`

These operations have precise semantics:

- **cache.read**: loads a tensor from the cache table at the given index and
  writes it into a normal graph variable. The output must be declared with the
  same dtype and shape as the table entries.
- **cache.write**: stores a tensor into the cache table at the given index. The
  input tensor must match the table dtype and shape.
- **cache.increment / cache.decrement**: update an index variable. These indices
  are typically stored as persistent scalars and are used as the table indices.
- **cache.reset**: clears the cache table or resets a specific index variable,
  depending on the arguments. Use reset when you need a clean state, for example
  when processing a new sequence.

Because these are nodes, they appear in traces and validation errors. This is
important for debugging. If you see an incorrect cache access, you can trace
back to the exact node that updated the index.

Ordering and barriers
---------------------

Cache operations often involve side effects that are not captured by data edges.
For example, you may increment an index and then read from the table using that
index. While the graph is explicit, you still need to ensure ordering is clear.
Use `barrier` or `dep` to enforce ordering when necessary.

Example:

.. code-block:: rust

   cache.increment 1 t;
   barrier;
   cache.read K[layer, t] >> k_out;

Without the barrier, a future scheduler might reorder the increment and read.
OpenInfer currently executes nodes in order within a block, but the explicit
barrier makes your intent unambiguous and future-proofs your graph if the
runtime gains more aggressive scheduling.

Cache validation rules
----------------------

Validation ensures cache correctness before execution:

- Index variables must exist and be scalar.
- Table indices must match the declared table index list.
- The output/input tensor dtype and shape must match the table entry dtype and
  shape.
- `@auto_dim` tables require the index variables to exist and be numeric.

These rules prevent subtle bugs like writing a `f32` tensor into an `f16` cache
or using the wrong index order.

Debugging cache issues
----------------------

When cache behavior looks wrong, follow this workflow:

1. Enable tracing (`OPENINFER_TRACE=full`) to capture cache nodes.
2. Inspect the trace for `cache.*` events and their order.
3. Verify index variables and their values at each step.
4. Ensure `cache.increment` or `cache.reset` nodes are placed correctly.
5. Check that you are not reusing a stale index across blocks.

Because the cache is explicit, most bugs are traceable to a missing increment,
a wrong index variable, or a mismatch between table declarations and usage.

Practical example: KV cache
---------------------------

Consider a transformer KV cache with `L` layers and a growing time dimension.
You can model it as:

.. code-block:: rust

   persistent {
     t: i32 @init(0);
     K(l, t): f16[H, Dh] @auto_dim(l, t);
     V(l, t): f16[H, Dh] @auto_dim(l, t);
   }

   block entry {
     cache.increment 1 t;
     loop layers (l in 0..L) {
       op compute_k(x, l) >> k;
       op compute_v(x, l) >> v;
       cache.write k >> K[l, t];
       cache.write v >> V[l, t];
     }
     return;
   }

This encodes the cache update directly. Each step increments time `t` and writes
the new key/value tensors. There is no hidden state; the graph tells the full
story.

Slice indices and rolling windows
---------------------------------

A common pattern is a rolling window over recent states. For example, you might
want to keep the last `W` entries of a cache and ignore older entries. OpenInfer
does not provide implicit slicing. Instead, you express the slice by controlling
the index variables explicitly.

Here is one approach using a modulo index to maintain a circular buffer:

.. code-block:: rust

   persistent {
     t: i32 @init(0);
     K(l, t): f16[H, Dh] @auto_dim(l, t);
   }

   block entry {
     cache.increment 1 t;
     op mod(t, W) >> t_mod;
     cache.write k >> K[l, t_mod];
     return;
   }

This pattern keeps a fixed window of size `W`. The table is still auto-dim, but
you only write to indices within `[0, W)`, which effectively turns it into a
ring buffer. This is a good example of explicit control: you decide the indexing
strategy rather than relying on runtime heuristics.

Cache lifecycle and reset strategies
------------------------------------

Persistent state must be managed. If you reuse a graph across multiple inputs,
you must decide when to reset or reuse caches. OpenInfer gives you the tools to
do this explicitly:

- Use `cache.reset` at the start of a run to clear tables.
- Use `cache.reset` or `cache.decrement` to roll back indices.
- Keep index variables (`t`, `rows`, `cols`) in persistent memory so they
  survive between steps.

For example, to process independent sequences in a batch, you might reset
indices at the start of each sequence:

.. code-block:: rust

   block entry {
     cache.reset t;
     cache.reset K;
     // process new sequence...
     return;
   }

The important idea is that cache lifecycle is explicit and testable. If a cache
is too large or too small, you can see the nodes that update it.

Performance considerations
--------------------------

Cache tables can become large quickly, especially with auto-dim growth. A few
practical tips:

- Keep table entries small and fixed in shape.
- Use packed dtypes where possible to reduce bandwidth.
- Avoid unnecessary cache writes; each write is a memory operation.
- Use barriers only when needed to avoid blocking op execution.

If performance is a concern, enable tracing and measure the time spent in cache
operations. Because caches are explicit nodes, you can quantify their cost and
decide whether to restructure your graph.

Where to go next
----------------

- `Control Flow` for branches, loops, and barriers.
- `Memory Model` for memory kinds and lifetime rules.
- `Adding Ops` for how new ops can be used with cache patterns.
