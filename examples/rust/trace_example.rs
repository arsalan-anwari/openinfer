use openinfer::{graph, insert_executor, Device, ModelLoader, Simulator};
use rand::Rng;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/minimal_model.oinf");
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

    let sim = Simulator::new(&model, Device::Cpu)?
        .with_trace()
        .with_timer();
    let mut exec = sim.make_executor(&g)?;

    let mut rng = rand::thread_rng();
    let len = model.size_of("B")?;
    let input: Vec<f32> = (0..len)
        .map(|i| {
            let base = rng.gen_range(-10.0..=10.0);
            base + (i as f32 * 0.001)
        })
        .collect();
    insert_executor!(exec, { x: input });

    exec.step()?;

    let trace = exec.trace();
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/rust/out");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("trace_example.json");
    std::fs::write(out_path, serde_json::to_string_pretty(&trace)?)?;

    Ok(())
}
