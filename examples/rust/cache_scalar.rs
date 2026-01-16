use openinfer::{fetch_executor, graph, ModelLoader, Simulator};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/cache_scalar_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        volatile {
            out_step: i32;
        }

        persistent {
            step: i32 @init(0);
        }

        block entry {
            cache.read step >> out_step;
            cache.increment step;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    exec.step()?;
    fetch_executor!(exec, { out_step: i32 });
    println!("step 0 read = {}", out_step);

    exec.step()?;
    fetch_executor!(exec, { out_step: i32 });
    println!("step 1 read = {}", out_step);

    Ok(())
}
