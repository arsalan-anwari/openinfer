use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/loop_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D, 3*D];
        }

        volatile {
            y: f32[D, 3*D] @init(0.0);
        }

        constant {
            QKV(layer, head): f32[D, 3*D] @pattern("attn.{head}.qkv.{layer}");
        }

        block entry {
            op add(x, QKV[0, 0]) >> y;

            loop layers (l in 0..num_layers) {
                loop heads (h in 0..num_heads) {
                    op add(y, QKV[l, h]) >> y;
                }
            }

            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?.with_trace();
    let mut exec = sim.make_executor()?;

    let d = model.size_of("D")?;
    let len = d * 3 * d;
    let input = Random::<f32>::generate_with_seed_opts(
        42,
        (-1.0, 1.0),
        len,
        TensorOptions {
            shape: Some(vec![d, 3 * d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    openinfer::log!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
