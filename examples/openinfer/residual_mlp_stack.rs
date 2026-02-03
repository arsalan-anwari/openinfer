use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/residual_mlp_stack.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            bias0: f32[D];
            W(layer): f32[D, D] @pattern("res.mlp.w.{layer}");
            b(layer): f32[D] @pattern("res.mlp.b.{layer}");
        }

        volatile {
            y: f32[B, D];
            h: f32[B, D];
        }

        block entry {
            op add(x, bias0) >> y;
            loop layers (l in 0..num_layers) {
                op matmul(y, W[l]) >> h;
                op add(h, b[l]) >> h;
                op relu(h, alpha=0.0) >> h;
                op add(y, h) >> y;
            }
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        1,
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
