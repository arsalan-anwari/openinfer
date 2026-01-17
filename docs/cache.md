# Persistent Memory (a.k.a. "Cache")

Some inference models require **persistent memory across steps**:

* Transformer KV cache
* Recurrent hidden state
* Streaming buffers
* Rolling windows

OpenInfer models this using `persistent` memory, a generic persistent storage abstraction.

## Declaring a Cache

```rust
persistent {
    step: i32 @init(0);
    cache: f16[H, H];
}
```

`@init(...)` literals must match the declared dtype: float literals for `f16/f32/f64`,
integer literals for integer/bool/bitset types. For example, `@init(5)` is invalid
for `f32`; use `@init(5.0)` instead.

Properties:

* Cache lives outside `block entry`
* Cache persists across executions
* Cache is architecture-agnostic

## Prefixed cache

Just like a prefix table in `volatile` and `constant`, you can create a table layout for `persistent` memory, which you can access using one or more indices.

This will essentially make a `n` dimensional table for any tensor layout. The table can either be fixed size (for the indices of `n`) or it can dynamically grow.

This depends on the attributes you set. By default prefix cache will be dynamic.

See examples below.

```rust
persistent {
    A(i): f32[D] @table;
    B(i, j): f32[D] @table;
    C(i): f16[D, H] @table;
    D(i, j): f16[D, H] @table @fixed(i=1024, j=256);

    // Example of KV[H, Dh] matrix for each attention head and token.
    K(l, t): f16[H, Dh] @table;
    V(l, t): f16[H, Dh] @table;
}
```

- `A`: A growable 1D table with `f32[i * D]` elements, accessed like `A[0..i] -> f32[D]`.
- `B`: A growable 2D table with `f32[i * j * D]` elements, accessed like `B[0..i, 0..j] -> f32[D]`.
- `C`: A growable 1D table with `f16[i * D * H]` elements, accessed like `C[0..i] -> f16[D, H]`.
- `D`: A fixed size 2D table with `f16[1024 * 256 * D * H]` elements, accessed like `D[0..1023, 0..255] -> f16[D, H]`.

### Indices slice access

You can also access a slice of the table entries.

For example lets say in a previous step you used `A[10]`, then the prefix cache will contain a table of size `f32[10, D]` (10 columns of D rows). Then in the current step you can access slices of this table like:

- `A[] -> f32[10, D] == A[0..9]`
- `A[0..5] -> f32[5, D]`
- `A[2..5] -> f32[3, D]`
- `A[0..-3] -> f32[7, D] == A[0..6]`

## Autodim cache

In some instances its preferable to have a matrix with an initial fixed size dimension which can grow dynamically across inference steps. For example in modern LLMs the `Key` and `Value` matrices from previous steps are reused so they do not need to be recomputed. This means new weights are appended as new columns and rows of the existing matrices.

OpenInfer implements this as a persistent cache using a special attribute named `@auto_dim({indices})`. You specify cache indices which are mapped to the tensor dimensions. The cache grows when you access it with scalar indices that exceed the current auto-dim counts. The current count for each auto-dim index can be referenced by name when slicing, but slice reads do not grow the cache.

You can still set the max size of these indices using `@fixed`, and you can optionally combine `@auto_dim` with a regular table layout.

See examples below:

```rust
persistent {
    A(i, j): f32[D, H] @auto_dim(i, j);
    B(i, j): f32[D, H] @auto_dim(i, j) @fixed(i=1024, j=256);
    C(l, i, j): f32[D, H] @table @auto_dim(i, j);
}
```

- `A`: A growable 2D matrix with `f32[(D + i) x (H + j)]` elements. When accessed with scalar indices like `A[i, j]`, the auto-dim counts grow to at least `i` and `j` and the full matrix is returned.

- `B`: Same as `A` but the matrix is bounded to maximum size `[D+1024, H+256]`.

- `C`: A growable 1D table containing a 2D matrix of size `f32[l * (D + i) * (H + j)]`. It has a similar access pattern as `A` but with an additional index `l` in the beginning like `C[l, i, j]`. The same rules for index slices apply here, so using `C[0..4, i, j]` returns a higher-rank tensor with `[4 * (D + i) * (H + j)]` elements.

### Growing the cache per step

To increase cache size per step, keep counters in persistent scalars and use them as the scalar indices that drive auto-dim growth.

```rust
persistent {
    rows: i32 @init(0);
    cols: i32 @init(0);
    M(r, c): f16[D, H] @auto_dim(r, c);
}

block entry {
    assign out: f16[D, H];

    cache.increment 3 rows;
    cache.increment 5 cols;

    // Access with scalar indices updates auto-dim counts and grows M.
    cache.read M[rows, cols] >> out;
    return;
}
```

## Cache Operations

Cache access is explicit and side-effectful.

```rust
cache.read  K[l, step] >> k;
cache.write v >> V[l, step];
cache.increment step;
cache.increment 5 step;
cache.decrement step;
cache.decrement 2 step;
cache.reset step;
cache.reset K;
cache.reset K[l];
cache.reset K[l, step];
```

Available primitives:

* `cache.read`
* `cache.write`
* `cache.increment`, `cache.increment {number}`
* `cache.decrement`, `cache.decrement {number}`
* `cache.reset`

This makes data dependencies and ordering explicit and analyzable.

## Example: Single Inference Step with Cache

```rust
graph! {

  dynamic {
    x: f32[B, D];
  }

  volatile {
    z: f32[B, D];
    W(l): f32[D, D] pattern("W.{l}");
  }

  persistent {
    step: i32 @init(0);
    K(l, t): f16[H, Dh] @table;
    V(l, t): f16[H, Dh] @table;
  }

  block entry {
    assign h: f32[B, D];
    assign k: f16[H, Dh];
    assign v: f16[H, Dh];

    transfer x >> h;

    loop layers (l in 0..10) {
      cache.read  K[l, step] >> k;
      cache.read  V[l, step] >> v;

      op attn(h, k, v, W[l]) >> h;

      cache.write k >> K[l, step];
      cache.write v >> V[l, step];
    }

    cache.increment step;

    return;
  }
}
```

> `transfer` is not guaranteed to be a deep copy, it can be pointer alias, reference or just reusing existing variable in `Synthesizer`.

## Multiple Steps in the Simulator

Running the graph multiple times advances the cache.

```rust
let mut sim = Simulator::new(&model, &g, Device::Cpu)?;
let exec = sim.make_executor()?;

insert_executor!(exec, { x: first_token });
exec.step()?;

insert_executor!(exec, { x: second_token });
exec.step()?;
```

Each invocation:

* Reuses cache contents
* Writes new entries at the next step
* Advances the cache cursor
