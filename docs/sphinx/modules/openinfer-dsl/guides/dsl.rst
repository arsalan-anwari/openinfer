DSL Guide (Complete Feature Set)
================================

This chapter is a comprehensive guide to the `graph!` DSL used in OpenInfer.
It is intentionally long and detailed because the DSL is the *primary* way you
author execution logic. The DSL is not a compiler and it does not perform hidden
optimizations. Instead, it expands into Rust code that constructs a runtime
`Graph` object. That `Graph` then drives the simulator and executor exactly as
declared. The benefit is that every control-flow decision, memory allocation,
and op selection is explicit. The trade-off is that the DSL syntax is strict and
you must express your intent directly.

The DSL has two top-level sections: memory declarations and blocks. Memory
declarations define the names, dtypes, dimensions, and storage kinds for every
variable you will use. Blocks define a sequence of nodes that the executor
follows. If you keep those two mental buckets in mind, the rest of the grammar
is mostly about expressing *which* nodes run *when*, and *which* variables they
read or write.

If you are new to OpenInfer, read this chapter after the `Architecture Overview`
and `Memory Model` guides. Those explain why the DSL exists and how it maps to
runtime data structures. This guide focuses on the complete feature set,
including cache operations, control flow, attributes, and the precise expansion
rules used by the procedural macro.

Mental model and structure
--------------------------

The DSL is a builder for a `Graph`. Every declaration you write turns into a
`VarDecl` or a `Node` inside the `Graph`. There is no separate optimization
stage that rewrites your graph. That means the graph is already in the form the
executor will traverse. When the simulator runs, it validates the graph against
the model and then the executor steps through nodes in the order dictated by
blocks, branches, and loops.

The DSL's top-level sections are:

1. **Memory sections**: `dynamic`, `volatile`, `constant`, `persistent`.
2. **Blocks**: `block entry { ... }` and any number of named blocks.

Memory declarations are always scoped to the whole graph. Blocks are named and
can be referenced by `branch` and `loop`. A block is a sequence of nodes. Nodes
include assignments, op calls, cache reads/writes, control flow, and sync
directives like `barrier` or `dep`.

The following example is a minimal but complete DSL program:

.. code-block:: rust

   let g = graph! {
     dynamic {
       x: f32[B, D];
     }
     constant {
       W: f32[D, D];
     }
     volatile {
       y: f32[B, D];
     }

     block entry {
       op matmul(x, W) >> y;
       return;
     }
   };

Even this small program illustrates important facts: memory comes first, the
block has a name (`entry`), and ops always write to a named output. That output
must have been declared in memory (or assigned in the block).

Because the DSL expands to Rust code, your `graph!` invocation can appear
anywhere in your program, including in library code. The expansion creates a
new `Graph` at runtime. It does not embed the graph as a static constant; it
creates it at runtime with normal Rust expressions. That makes it easy to reuse
configuration or to generate graphs programmatically, but it also means you
should expect runtime validation errors if you make a mistake in names, sizes,
or dtypes.

Memory declarations in depth
----------------------------

Memory declarations describe *what* variables exist, *where* they live, and
*how* they are indexed. OpenInfer uses memory kinds to make lifetimes and
mutability explicit. The DSL exposes this directly with four sections:

- `dynamic`: Inputs provided by the executor at runtime.
- `volatile`: Scratch buffers that reset each execution step.
- `constant`: Parameters loaded from the `.oinf` model.
- `persistent`: Mutable state that survives across steps.

Each variable declaration includes a name, dtype, and dimensions. Dimensions are
strings in the graph. They can be literal integers (`128`) or sizevar names
(`B`, `D`). Sizevars are resolved at runtime by reading values from the `.oinf`
header, which makes models flexible across sizes without rewriting the graph.

The simplest declaration is:

.. code-block:: rust

   dynamic {
     x: f32[B, D];
   }

You can add attributes to a variable using `@` syntax. Some attributes are
general (e.g., `@ref`) and others are specific to cache tables (`@table`,
`@auto_dim`, `@fixed`). A common pattern for constants is to bind a variable name
to a specific tensor in the model:

.. code-block:: rust

   constant {
     w1: f32[D, H] @ref("w1");
     b1: f32[H] @ref("b1");
   }

The `@ref` attribute sets the model-facing name used by validation and tooling.
If it is omitted, the simulator matches by the variable name. Runtime loading
currently looks up tensors by the variable name, so keep names aligned unless
you are using `@ref` purely for validation or documentation clarity.

Prefix tables allow a single declaration to map to a *family* of tensors. This
is used for multi-layer models where each layer has a similar tensor set:

.. code-block:: rust

   constant {
     W(l): f32[D, D] @pattern("W.{l}");
     B(l): f32[D] @pattern("B.{l}");
   }

Here the DSL declares a symbolic index `l` that the runtime uses to resolve
names such as `W.0`, `W.1`, and so on. The mapping is still explicit: it is not
magic, just a structured pattern that expands to concrete names at runtime.

Persistent variables can be marked as cache tables. Tables use symbolic indices
to represent access patterns. For example, a KV cache might be indexed by layer
and time:

.. code-block:: rust

   persistent {
     K(l, t): f16[H, Dh] @table;
     V(l, t): f16[H, Dh] @table;
   }

Tables can also be auto-dimensioned so they grow as the index grows. This is
useful for streaming workloads where you do not know the final sequence length:

.. code-block:: rust

   persistent {
     rows: i32 @init(0);
     cols: i32 @init(0);
     M(r, c): f16[D, H] @auto_dim(r, c);
   }

The auto-dim cache pattern requires explicit cache operations to update the
dimensions. You must increment the indices before accessing the table, which
makes growth explicit and easy to reason about. This pattern keeps stateful
allocation in the graph rather than hidden in the runtime.

Blocks and node types
---------------------

Blocks are named sequences of nodes. The executor starts at `entry` and follows
branches and loops. Each node has a type, and the node type determines both
validation rules and execution semantics. The DSL supports a fixed set of node
types; there are no user-defined nodes at this time.

The most common node types are `assign` and `op`. Assignments declare temporary
variables within a block, while ops perform computations. An `assign` is
explicitly typed and can include dimensions:

.. code-block:: rust

   assign h: f32[B, D];
   op matmul(x, W) >> h;

The op syntax is a function-like invocation with optional attributes. The output
is always declared after `>>`. If you omit `assign` for an output that does not
already exist in memory, the graph will be invalid. This is intentional: the
DSL wants you to declare outputs in memory or explicitly assign them as
temporaries, which keeps memory usage explicit.

Attributes are key-value pairs that are parsed into `AttrValue` in the runtime.
They can be numeric, boolean, dtype literals, or references. For example:

.. code-block:: rust

   op clamp(x, min=0.0, max=6.0) >> y;
   op cast(x, to=f16) >> y;
   op relu(x, alpha=0.0) >> y;

The attribute types are preserved and validated against the op schema. This
matters for ops like `fill`, which require the attribute value to match the
tensor dtype. If you pass an incompatible attribute, the simulator will reject
the graph before execution.

Cache operations are explicit node types. They are used to read or write
persistent tables, update indices, and manage rolling windows. The DSL exposes
them as `cache.read`, `cache.write`, `cache.increment`, `cache.decrement`, and
`cache.reset`. For example:

.. code-block:: rust

   cache.increment 1 rows;
   cache.read M[rows, cols] >> out;
   cache.write in >> M[rows, cols];

These nodes make memory effects explicit, which is crucial for reasoning about
stateful models. If you see a cache read in a trace, you know exactly where the
state is coming from. The runtime does not perform hidden caching.

Control flow nodes include `branch`, `loop`, `yield`, `await`, `barrier`, `dep`,
`transfer`, and `return`. Branches jump between blocks, loops repeat a block
with a named iterator, and yield/await coordinate execution across blocks. The
DSL keeps these as first-class node types so they are visible in the graph
structure and in traces.

One of the most important points is that control flow is *explicit* and *local*
to the graph. There is no hidden scheduler that reorders nodes for performance.
When you write:

.. code-block:: rust

   op matmul(x, W) >> h;
   barrier;
   op relu(h) >> h;

you are stating that the barrier must be respected. If you need ordering between
nodes that do not share data, you can use `dep` to enforce it. This is a powerful
tool for managing stateful side effects or external resources without modifying
op semantics.

Expansion and validation details
--------------------------------

The DSL is implemented as a procedural macro. Parsing happens using `syn` and
the AST is translated into `Graph` construction code. Understanding this
expansion is useful when debugging complex graphs or when contributing to the
DSL itself.

Parsing produces an intermediate representation that mirrors the DSL grammar.
Each memory declaration is converted into a `VarDecl`, and each node in a block
is converted into a `NodeKind` value. During expansion, the macro constructs a
`Graph` by calling `Graph::new()`, `Graph::add_var(...)`, `Graph::add_block(...)`,
and `Graph::add_node(...)` with the parsed values.

Because this expansion happens at runtime, errors are raised during simulation,
not at compile time. For example, if a sizevar name is misspelled, the simulator
will not find it in the `.oinf` file and will return an error. Similarly, if you
use an op with incompatible dtypes, validation will fail before execution.

This design is deliberate. It keeps the DSL simple and avoids tying it to
compiler plugins or external code generators. It also makes the graph fully
serializable, which is useful for debugging. You can print the graph as JSON and
inspect the node list, variable declarations, and control flow. The JSON
representation mirrors the runtime graph, so it is a faithful view of what will
execute.

For example, the following DSL fragment:

.. code-block:: rust

   block entry {
     assign t0: f32[B];
     op add(x, a) >> t0;
     op mul(y, t0) >> y;
     return;
   }

will produce a block with three nodes (`Assign`, `Op`, `Op`) followed by a
`Return`. Each node carries its index, UUID, and the associated op or variable
names. This is precisely what the executor sees, which is why OpenInfer traces
are useful: they can be correlated directly to DSL lines.

Debugging DSL graphs
--------------------

When a DSL graph fails to validate, the simulator error will indicate which
node or variable was problematic. The best debugging workflow is:

1. Serialize the graph to JSON.
2. Locate the node or variable by name.
3. Compare the DSL with the resolved graph.
4. Fix mismatched names, dims, or dtypes.

For quick iteration, you can temporarily minimize the graph: comment out blocks,
reduce the number of ops, and add `return` early. Since the DSL is deterministic,
this often isolates the error quickly.

If a graph validates but produces wrong results, enable tracing and compare the
trace to the expected execution order. The trace includes node indices, op
names, and block names, which helps you map runtime events back to the DSL.
Because the DSL is explicit about control flow, unexpected order almost always
implies a missing `barrier`, `dep`, or `await` in your graph.

Finally, remember that the DSL is intentionally strict. It forces you to encode
state transitions, memory usage, and control flow explicitly. This is a feature,
not a limitation. The clarity you gain is what makes debugging and extending
OpenInfer possible at scale.

Complete feature reference (by concept)
---------------------------------------

This section summarizes the full DSL feature set with additional context and
usage guidance. It is intentionally verbose so you can use it as a reference
without digging into the macro source.

**Memory kinds and attributes**

All variables must be declared in one of the memory sections. The memory kind is
part of the variable’s identity and drives validation rules:

- `dynamic` variables must be provided by the executor at runtime and are not
  mutated inside the graph.
- `volatile` variables are scratch buffers that are reset between steps.
- `constant` variables are loaded from the `.oinf` model and are read-only.
- `persistent` variables survive across steps and are mutable.

Attributes such as `@init`, `@ref`, `@pattern`, `@table`, `@fixed`, and
`@auto_dim` are stored in the `VarDecl` structure. They do not change the
variable’s dtype or shape, but they change how the runtime treats the variable.

**Assignments**

Assignments declare a new variable within a block. They are often used for
temporary buffers to store intermediate results. Because assignments are explicit,
you can reason about memory usage and lifetimes with high confidence. A common
pattern is to create a temporary buffer and reuse it across multiple ops within
the same block. This is cheaper than allocating a new buffer for every op, and
because it is explicit you can decide when it is safe.

**Op invocations**

Ops are the core compute nodes. They accept input variables and write to an
explicit output. The output must already exist in memory or be assigned earlier
in the block. This pattern keeps the graph’s memory usage explicit and makes
reuse possible. Op attributes are typed and validated. The attribute type
system is intentionally strict to prevent silent errors.

**Control flow**

Control flow nodes are not opaque. They are part of the graph. Use `branch` for
conditional jumps, `loop` for repeated execution, and `return` to end a block.
Use `yield`/`await` to coordinate concurrent blocks. Use `barrier` and `dep` to
enforce ordering. The DSL is designed so that all control flow can be reasoned
about by reading the graph JSON. This is a core principle of OpenInfer.

**Cache and tables**

Cache operations interact with persistent tables. This includes reading and
writing table entries, incrementing indices, and resetting state. Table indices
are explicit symbols that you manage directly. If you need slicing semantics,
express them explicitly by manipulating the index variables. This is more verbose
than automatic slicing, but it keeps the state machine visible and traceable.

Common patterns and pitfalls
----------------------------

The DSL’s explicitness is powerful but it also makes certain mistakes more
visible. The most common pitfalls are:

- **Missing declarations**: If a variable is used without a declaration, the
  graph is invalid. Always declare outputs explicitly.
- **Sizevar mismatches**: If a dimension name is misspelled, the runtime cannot
  resolve it. Keep sizevar names consistent between DSL and `.oinf`.
- **Attribute type mismatch**: Attributes are typed at parse time. If you pass
  an integer to an op expecting a float, the simulator will reject the graph.
- **Incorrect table indices**: Table indices must match the declaration order.
  Swapping indices leads to incorrect reads/writes.
- **Implicit ordering assumptions**: If you rely on a specific order between
  nodes without data edges, use `barrier` or `dep`.

To avoid these issues, follow these practices:

- Keep memory declarations close to the top of the DSL, and group by kind.
- Use explicit assignments for intermediate buffers.
- Use block names that mirror your algorithm phases.
- Add comments around cache updates to clarify intent.
- Validate often with the simulator and trace output.

When in doubt, serialize the graph to JSON. The JSON representation is a clear
map of what the runtime will execute. If the JSON is wrong, the runtime behavior
will be wrong as well. Treat it as your ground truth.
