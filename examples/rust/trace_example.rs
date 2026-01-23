use openinfer::{graph, insert_executor, ModelLoader, Random, Simulator};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            a: f32[B];
            y: f32[B] @init(5.0);
        }

        block entry {
            assign t0: f32[B];
            op add(x, a) >> t0;
            op mul(y, t0) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?
        .with_trace()
        .with_timer();
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;
    insert_executor!(exec, { x: input });

    exec.step()?;

    let trace = exec.trace();
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/rust/out");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("trace_example.json");
    std::fs::write(out_path, serde_json::to_string_pretty(&trace)?)?;

    Ok(())
}
