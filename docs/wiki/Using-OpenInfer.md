# Using OpenInfer

This page shows the typical workflow at a high level.

## 1) Load a model

```rust
use openinfer::ModelLoader;

let model = ModelLoader::open("openinfer-simulator/res/models/mlp_regression.oinf")?;
```

## 2) Define a graph

```rust
use openinfer::graph;

let g = graph! {
    dynamic { x: f32[B, D]; }
    constant { w1: f32[D, H]; b1: f32[H]; }
    volatile { y: f32[B, H]; }

    block entry {
        op matmul(x, w1) >> y;
        op add(y, b1) >> y;
        return;
    }
};
```

## 3) Run a step and fetch results

```rust
use openinfer::{fetch_executor, insert_executor, Simulator, Tensor};

let sim = Simulator::new(&model, &g, Device::Cpu)?;
let mut exec = sim.make_executor()?;

insert_executor!(exec, { x: input_tensor });
exec.step()?;

fetch_executor!(exec, { y: Tensor<f32> });
```

## 4) Optional tracing and timing

Enable tracing and timers when you need to inspect execution:

```rust
let sim = Simulator::new(&model, &g, Device::Cpu)?
    .with_trace()
    .with_timer();
```

## Next steps

- Learn how memory kinds work: [Core Concepts](Core-Concepts)
- See supported ops and dtypes: [Capabilities](Capabilities)
