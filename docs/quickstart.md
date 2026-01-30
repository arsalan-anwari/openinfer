# Quickstart

## Minimal Example

A minimal graph that:

* Takes an input tensor
* Applies two operations
* Returns a result

> Example of `model.oinf` file in non binary format.
```ini
B := 1024
a: f32[B] = {3.4324, 53.24324, 2334.2345 ...}
```

> minimal.rs
```rust
use openinfer::{
    graph, fetch_executor, insert_executor, Device, ModelLoader, Random, Simulator, Tensor,
};

fn main() -> anyhow::Result<()> {
    let model = ModelLoader::open("res/models/minimal_model.oinf")?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            a: f32[B];
            y: f32[B] @init(5.0);
        }

        block entry {
            assign t0: f32[B];
            op add(x, a) >> t0;
            op mul(y, t0) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, Device::Cpu)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    println!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
```

Variables like `[B]` are named sizes, which are defined in the `.oinf` model; these can be dynamic. The `Simulator` validates that dimensions and dtypes are consistent with the model.

The variables defined in the model binary and the DSL do not need to be exactly the same. The DSL can have new variables not found in the binary, but the DSL cannot have the same variable name with a different data type or dimension. By default the variables are linked between the binary and DSL. So `a: f32[B]` in the DSL is directly linked to `a: f32[B]` in the binary by default.

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
