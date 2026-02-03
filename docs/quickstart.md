# Quickstart

## Minimal Example

A minimal graph that:

* Takes an input tensor
* Applies two operations
* Returns a result

> Conceptual text view of a `.oinf` model (as printed by `verify_oinf.py`).
```ini
B := 1024
a: f32[B] = {3.4324, 53.24324, 2334.2345 ...}
```

> mlp_regression.rs
```rust
use openinfer::{
    fetch_executor, graph, insert_executor, Device, ModelLoader, Random, Simulator, Tensor,
    TensorOptions,
};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("res/models/mlp_regression.oinf")?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            w1: f32[D, H];
            b1: f32[H];
            w2: f32[H, O];
            b2: f32[O];
        }

        volatile {
            h: f32[B, H];
            y: f32[B, O];
        }

        block entry {
            op matmul(x, w1) >> h;
            op add(h, b1) >> h;
            op relu(h, alpha=0.0) >> h;
            op matmul(h, w2) >> y;
            op add(y, b2) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, Device::Cpu)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let input = Random::<f32>::generate_with_seed_opts(
        0,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    println!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
```

Variables like `[B]` are named sizes, which are defined in the `.oinf` model; these can be dynamic. The `Simulator` validates that dimensions and dtypes are consistent with the model.

The variables defined in the model binary and the DSL do not need to be exactly the same. The DSL can have new variables not found in the binary, but the DSL cannot have the same variable name with a different data type or dimension. By default the variables are linked between the binary and DSL. So `w1: f32[D, H]` in the DSL is directly linked to `w1: f32[D, H]` in the binary by default.

## Executor Macros

These macros bridge between user data and the executor. Use the panic-on-error versions for quick scripts, and the `try_*` versions when you want to handle errors yourself.

```rust
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
insert_executor!(exec, { x: x });
fetch_executor!(exec, { y: Tensor<f32> });
println!("y = {:?}", y.data);
fetch_executor!(exec, { alpha: f32 });
println!("alpha = {}", alpha);
```

```rust
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0])?;
try_insert_executor!(exec, { x: x })?;
let y: Tensor<f32> = try_fetch_executor!(exec, { y: Tensor<f32> })?;
println!("y = {:?}", y.data);
```

```rust
let (y, z) = (
    try_fetch_executor!(exec, { y: Tensor<f32> })?,
    try_fetch_executor!(exec, { z: i64 })?,
);
println!("y = {:?}, z = {:?}", y.data, z);
```

> The `*_fetch_*` macros support optional type hints. If you omit the type, Rust will try to infer it from usage. Scalars (no dims) return native values like `f32` instead of `Tensor<f32>`.
