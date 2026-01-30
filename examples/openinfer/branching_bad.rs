use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/branching_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            w: f32[D, D];
        }

        volatile {
            h: f32[B, D];
            cond: bool @init(false);
        }

        block entry {
            op matmul(x, w) >> h;
            op is_finite(h) >> cond;
            branch cond ok bad;
            branch algorithm;
            return;
        }

        block ok {
            op relu(h, alpha=0.0) >> h;
            return;
        }

        block bad {
            op fill(x, value=0.0) >> x;
            op fill(h, value=0.0) >> h;
            return;
        }

        block algorithm {
            op add(h, x) >> h;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?.with_trace().with_timer();
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let mut input = Random::<f32>::generate_with_seed_opts(
        3,
        (-2.0, 2.0),
        b * d,
        openinfer::TensorOptions {
            shape: Some(vec![b, d]),
            ..openinfer::TensorOptions::default()
        },
    )?;
    if !input.data.is_empty() {
        input.data[0] = f32::NAN;
    }

    insert_executor!(exec, { x: input });

    exec.step()?;

    fetch_executor!(exec, { h: Tensor<f32>, cond: bool });
    openinfer::log!("branch condition: {}", cond);
    openinfer::log!("h[0..100] = {:?}", &h.data[..100.min(h.len())]);

    let log = exec.trace();
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/openinfer/out");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("branching_bad_log.json");
    std::fs::write(out_path, serde_json::to_string_pretty(&log)?)?;

    Ok(())
}
