use openinfer::{fetch_executor, graph, insert_executor, Device, ModelLoader, Simulator};
use rand::Rng;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/relu_model.oinf");
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

    let sim = Simulator::new(&model, Device::CpuAvx2)?;
    let mut exec = sim.make_executor(&g)?;

    let mut rng = rand::thread_rng();
    let len = model.size_of("B")?;
    let input: Vec<f32> = (0..len)
        .map(|i| {
            let base = rng.gen_range(-3.0..=3.0);
            base + (i as f32 * 0.01)
        })
        .collect();

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: f32, negative_slope: f32, clamp_max: f32 });
    println!(
        "relu settings: negative_slope={}, clamp_max={}",
        negative_slope.data[0], clamp_max.data[0]
    );
    println!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
