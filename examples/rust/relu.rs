use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/relu_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            y: f32[B];
        }

        constant {
            negative_slope: f32;
            clamp_max: f32;
        }

        block entry {
            op relu(x, negative_slope=negative_slope, clamp_max=clamp_max) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-3.0, 3.0), len)?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32>, negative_slope: f32, clamp_max: f32 });
    println!(
        "relu settings: negative_slope={}, clamp_max={}",
        negative_slope, clamp_max
    );
    println!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
