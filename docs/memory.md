# Inputs and Outputs

Data used for the model should always defined at the top, before `block entry {}`.

```rust
dynamic {
    x: f32[B];
}

volatile {
    a: f32[B];
    y: f32[B];
}

constant {
    alpha: f32;
}

persistent {
    cache: f16[H, Dh];
}
```

There are 4 different types of memory that can used with the model:
1. `dynamic`: This is memory that can be mutated by an external program and is cleared every inference step.
    - This is usefull to feed things like: user input, tokens, weights, images and other data that gets generated on the fly.
2. `volatile`: This is memory that can be mutated from within a block of logic in the DSL and is filled with the content of the model binary file. Data is reset every inference step.
    - This is essentially the memory used for things like weights, tensors, etc in the binary file which can be used for performing calculations with the ops.
3. `constant`: This is similar to `volatile` where data is copied from the model binary, however this memory cannot be mutated, only read.
    - This is usefull for things like metadata, settings, op params, etc.
4. `persistent`: This is memory which can be mutated from within a block of logic in the DSL but stays peristant every inference step. This means you can store values from previous inference steps as history.
    - This is usefull for things like KV-cache, rolling windows, recurrent hidden state and anything that needs to persist for every inference step.

## Interacting with memory

- You can only mutate `dynamic` memory before the inference step is started or has completed. You can use the macro `insert_executor!{}` to modify one or more variables defined in the DSL with new data.
- You are free to fetch data during any time of the inference step that is stored in { `volatile`, `constant` or `persistent` }, but you **cannot** mutate it. You can use the macro `fetch_executor!{}` to get a copy of one or more variables defined in the DSL.

## Output reuse and memory pressure

Some ops (notably accumulation variants) can reuse an existing output buffer if
the output tensor already exists and matches dtype/shape. This reduces memory
churn and helps large graphs run within tight memory budgets. See
`docs/memory_reuse.md` for details and trade-offs.

## Attributes on variable definitions

Atrributes can be used only on variable definitions for things like linking model data, expressing layouts, quantization, or other metadata relevant for the Simulator and Synthesizer.

```rust
constants {
  alpha: f32 @ref("alpha");
  beta:  f32 @ref("beta");
  bias:  f32 @ref("gamma");
}
```

> For example you can use the `@ref` attribute to link a custom variable name in the DSL to a variable name in the binary.

## Operator settings are named parameters

Operations do **not** use attributes. Instead, operator configuration is expressed via **named parameters**.

```rust
op relu(h, negative_slope=0.0, clamp_min=0.0, clamp_max=inf) >> h;
```

Examples of realistic activation settings you might see in real deployments:

```rust
// Standard ReLU
op relu(h, negative_slope=0.0) >> h;

// LeakyReLU (common in CNNs)
op relu(h, negative_slope=0.01) >> h;

// Clipped ReLU / ReLU6 (common in mobile / quantization-aware)
op relu(h, negative_slope=0.0, clamp_max=6.0) >> h;

// Lower clamp (occasionally used for numerical stabilization)
op relu(h, negative_slope=0.0, clamp_min=-1e-6) >> h;
```

## What attributes mean

* Attributes are **declarative**: they restrict or guide Simulator and Synthesizer decisions.
* They do not guarantee a specific implementation.
* Unknown attributes can be preserved (for tooling) or rejected (for strict mode).

## Prefix Tables

Many models store repeated tensors under a predictable naming scheme, for example:

* `W.0`, `W.1`, ..., `W.9`
* `attn.qkv.0`, `attn.qkv.1`, ...

A **prefix table** declares a *family* of model tensors under one DSL name, indexed by one or more symbolic variables.

```rust
volatile {
  W(l): f32[D, D] @pattern("W.{l}");
}
```

How it works:

* `W(l)` declares an indexed handle `W[<expr>]` usable inside blocks.
* The attribute `@pattern("W.{l}")` tells the loader how to map an index `l` to a model key.
* Prefix tables are **declarations**: the graph references them, and the runtime resolves them from the model package.

You can also alias different naming schemes:

```rust
constant {
  QKV(layer, head): f16[D, 3*D] @pattern("attn.{head}.qkv.{layer}");
}
```

Prefix tables can **only** be defined in `volatile` and `constant` memory space.
