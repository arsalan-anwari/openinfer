use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/online_weight_update.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
            delta: f32[D, O];
        }

        constant {
            zero: f32[D, O];
        }

        volatile {
            w: f32[D, O];
            y: f32[B, O];
        }

        persistent {
            W: f32[D, O] @init(0.0);
        }

        block entry {
            cache.read W >> w;
            op add(w, zero) >> w;
            op matmul(x, w) >> y;
            op add(w, delta) >> w;
            cache.write w >> W;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let o = model.size_of("O")?;
    let x = Random::<f32>::generate_with_seed_opts(
        5,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;
    let delta = Random::<f32>::generate_with_seed_opts(
        6,
        (-0.1, 0.1),
        d * o,
        TensorOptions {
            shape: Some(vec![d, o]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: x.clone(), delta: delta.clone() });
    exec.step()?;
    fetch_executor!(exec, { y: Tensor<f32> });
    openinfer::log!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    insert_executor!(exec, { x: x, delta: delta });
    exec.step()?;
    fetch_executor!(exec, { y: Tensor<f32> });
    openinfer::log!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
