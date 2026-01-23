use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/worst_case_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D];
        }

        volatile {
            w: f32[D];
        }

        block entry {
            loop steps (l in 0..num_layers) {
                op add(x, w) >> x;
            }
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?
        .with_trace()
        .with_timer()
        .with_inplace(); // Disable this for true worst case

    let mut exec = sim.make_executor()?;

    let d = model.size_of("D")?;
    let input = Random::<f32>::generate_with_seed_opts(
        7,
        (-1.0, 1.0),
        d,
        TensorOptions {
            shape: Some(vec![d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { x: Tensor<f32> });
    println!("x.len() = {}", x.len());
    println!("x[0..100] = {:?}", &x.data[..100.min(x.len())]);
    Ok(())
}
