# Core Concepts

OpenInfer centers on an explicit inference graph defined in Rust. You describe
**what happens** and **in what order**, and the runtime executes it deterministically.

## Model package (.oinf)

Models live in a single `.oinf` file that stores:

- Size variables (e.g. `B`, `D`)
- Metadata (scalars, strings, arrays)
- Tensors (weights, constants, optional data)

These are loaded lazily and resolved by name when the graph executes.

## Memory kinds

The DSL has four memory sections:

- `dynamic`: provided per step by the caller (inputs)
- `volatile`: mutable per step, reset each run
- `constant`: read‑only, model‑backed values
- `persistent`: state that survives across steps (KV cache, rolling state)

## Blocks and control flow

Graphs are made of blocks with explicit nodes:

- Ops (`op add(...) >> out`)
- Assignments (`assign t: f32[B]`)
- Control flow (`branch`, `loop`)
- Effects (`cache.read`, `cache.write`, `barrier`, `dep`)
- Concurrency (`yield` / `await`)

This makes the execution order visible and traceable.

## DSL details and examples

### Memory sections

```rust
dynamic { x: f32[B, D]; }
constant { w: f32[D, O]; }
volatile { y: f32[B, O]; }
persistent { step: i32 @init(0); }
```

### Ops and attributes

```rust
op matmul(x, w) >> y;
op relu(y, alpha=0.0, clamp_max=6.0) >> y;
```

Attributes are named parameters and are validated by type.

### Control flow

```rust
block entry {
  assign cond: bool;
  op is_finite(y) >> cond;
  branch cond ok bad;
  return;
}

block ok { return; }
block bad { return; }
```

### Loops

```rust
loop layers (l in 0..num_layers) {
  op matmul(h, W[l]) >> h;
}
```

### Cache access (persistent state)

```rust
cache.read  K[l, step] >> k;
cache.write v >> V[l, step];
cache.increment step;
```

### Barriers, deps, and transfer

```rust
barrier;
dep after(matmul) before(cache.write);
transfer x >> h;
```

### Yield and await (concurrency)

```rust
yield x;
await x;
```

Yield releases a variable from the entry block so another block can consume it.
Await waits for all consumers to release it back.

## Execution model

You build a graph, create a simulator, then create an executor:

1. Validate graph against the model
2. Execute on CPU or Vulkan
3. Fetch outputs and optional traces

The simulator focuses on correctness and debuggability rather than raw speed.
