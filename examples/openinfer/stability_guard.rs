use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/stability_guard.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            bias: f32[D];
        }

        volatile {
            y: f32[B, D];
            cond: bool @init(true);
        }

        block entry {
            op is_finite(x) >> cond;
            branch cond ok bad;
            return;
        }

        block ok {
            op add(x, bias) >> y;
            return;
        }

        block bad {
            op fill(y, value=0.0) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        7,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: x });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32>, cond: bool });
    openinfer::log!("cond = {}", cond);
    openinfer::log!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
