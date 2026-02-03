use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/mlp_regression.oinf");
    let model = ModelLoader::open(model_path)?;

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

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        0,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: x });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    openinfer::log!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
