use openinfer::{graph, insert_executor, ModelLoader, Simulator};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_auto_dim_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            bias: f32;
        }

        volatile {
            out: f32[D, H];
        }

        persistent {
            rows: i32 @init(0);
            cols: i32 @init(0);
            M(r, c): f32[D, H] @auto_dim(r, c) @fixed(r=4, c=4);
        }

        block entry {
            cache.increment 5 rows;
            cache.increment 5 cols;
            cache.read M[rows, cols] >> out;
            op add(out, bias) >> out;
            cache.write out >> M[rows, cols];
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    insert_executor!(exec, { bias: 1.0f32 });
    match exec.step() {
        Ok(_) => {
            openinfer::log!("Unexpected success: cache access should exceed @fixed limits.");
        }
        Err(err) => {
            openinfer::log!("Expected cache limit error: {err}");
        }
    }

    Ok(())
}
