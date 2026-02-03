use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/moe_routing.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
            route: bool;
        }

        constant {
            w_gate: f32[D, 2];
            w0: f32[D, O];
            b0: f32[O];
            w1: f32[D, O];
            b1: f32[O];
        }

        volatile {
            gate: f32[B, 2];
            y: f32[B, O];
        }

        block entry {
            op matmul(x, w_gate) >> gate;
            branch route expert0 expert1;
            return;
        }

        block expert0 {
            op matmul(x, w0) >> y;
            op add(y, b0) >> y;
            return;
        }

        block expert1 {
            op matmul(x, w1) >> y;
            op add(y, b1) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        3,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: x, route: true });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    openinfer::log!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
