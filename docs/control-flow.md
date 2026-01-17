# Blocks and Execution

Execution logic is written inside **blocks**.

```rust
block entry {
    assign a: f32[B, D];
    op matmul(x, w) >> a;
    op add(a, bias) >> a;
    return;
}
```

Key properties:

* Blocks execute sequentially
* Each line represents a graph node
* Execution order is explicit
* Blocks end with a terminator (`return`, or `yield`)

Blocks form a **control-flow graph**, not just a flat list of ops.

## Assignments and Operations

* `assign` declares a temporary tensor or scalar stored in `volatile`.
* `op` executes a computation and produces an output
* output is alwasy stored in the variable following `>>`.

```rust
assign h: f32[B, D];
op matmul(x, w) >> h;
```

Assignments are **ephemeral**:

* They exist only during execution
* The runtime may reuse or alias memory
* They do not persist across steps
* The synthizier might optimize them out or use them as reference instead of deep copies.

## Loops

Loops are explicit control-flow constructs.

```rust
loop layers (l in 0..num_layers) {
    op matmul(h, W[l]) >> h;
    op relu(h) >> h;
}
```

> Here `layers` is just a name to identify it as a block in the graph; you can use any name like `heads`, `batches`, etc.

Characteristics:

* Loop bounds are symbolic
* Loop indices are explicit variables
* Loop bodies form nested regions
* Repetition is visible to the Synthesizer

## Barriers and Control Dependencies

Inference graphs often need explicit ordering boundaries for correctness, debugging, or interoperability.

### Barrier

A `barrier;` prevents motion, fusion, or reordering across the boundary.

```rust
block entry {
  assign h: f32[B, D];

  op matmul(x, W[0]) >> h;
  barrier;
  op relu(h) >> h;
  return;
}
```

### Explicit control dependency

A control dependency expresses ordering **without creating a data edge**.

This is useful when:

* you must enforce "write happens after compute" even if the value isn't used later
* you need deterministic traces
* you interface with an external effect that the Synthesizer must not reorder

Example: ensure a cache write happens after an op, but the output tensor itself is not otherwise consumed.

```rust
block entry {
  assign h: f32[B, D];

  op matmul(x, W[0]) >> h;

  // Record the computed activation into a cache table for debugging/inspection.
  // The `dep` makes the ordering explicit even though the write is an effect.
  dep after(matmul) before(cache.write);
  cache.write h >> K[0, step];

  op relu(h, negative_slope=0.0) >> h;
  return;
}
```

> The syntheziser is free to reorder pure ops, but it must respect explicit deps around effects.

## Branching and Yielding Across Blocks

Graphs are **control-flow graphs**. Here `entry` is always the starting block, but execution can jump to other blocks.

Only the entry block can assign variables to be used. All subblocks can mutate it, but cannot return value back to the entry block. Essentially all variables are globals.

> This is to make parsing and traversing the graph easier.

### Branch

Use `branch` to jump to another block (optionally based on a condition).

```rust
block entry {
  assign h: f32[B, D];
  assign cond: bool;

  op matmul(x, W[0]) >> h;
  op is_finite(h) >> cond;

  branch cond ok bad;
  branch algorithm;
  return;
}

block ok {
  op relu(h, negative_slope=0.0) >> h;
  return;
}

block bad {
  op fill(h, value=0.0) >> h;
  return;
}

block algorithm {
  // Some sequence of ops changing h...
  return;
}
```

### Yield

Use `yield {var}` when you want the entry block to remove temporary access to a variable.

This is useful for async or streaming execution.

After yielding, the entry block cannot mutate the variable used by the consuming blocks. However it is free to execute other code.

Using `await {var}`, multiple blocks can consume the same variable, but only one can mutate it.

Entry block has access to the variable whenever all consumers yield the variable.

```rust
block entry {
  assign h: f32[B, D];
  assign x: i32[D];
  assign h2: f32[B, D];

  op matmul(x, W[0]) >> h;
  yield x; //x not available anymore to entry

  // These ops are executed in parallel
  op relu(h, negative_slope=0.0, clamp_max=6.0) >> h;

  // Waiting for all consumers to be done.
  await x;
  // do something with x modified by consumer blocks...
  return h;
}

// A different device, core or thread could execute this
// The exact scheduling model is backend-defined.

block consumer_1 {
  await x;
  // some compute modifiying x.
  yield x;
}

block consumer_2 {
  await x;
  op relu(x, negative_slope=0.0, clamp_max=6.0) >> h2;
  yield x;
}

block consumer_3 {
  await x;
  // some compute reading x.
  yield x;
}
```

Notes:

* For sub blocks `yield` is a terminator like `return`. For the entry block its an invokation.
* It defines an explicit control-flow edge to a continuation block.
* Backends may interpret `yield` as "pause and resume", "send to a queue", or "schedule on another device". Implementation depends on device.
* Each consumer will have the last known value of the variable that was yielded by the entry block. This means that the consumer which mutates x will not affect other consumers which read x.
* Yield/await is supported on CPU and Vulkan backends; other backends reject it during validation.
* The runtime requires an `await` to complete a yield phase before the entry block can yield again.

> You can `yield` and `await` multiple variable like: `yield a, b, c;` and `await a, b, c;`. In this case the `await` will be serialized until all variables are available. This also means multiple blocks can mutate different variables, but the rule of 1 block per variable still applies. So for example `b1` mutates `a` and `b2` mutates `b, c`.
