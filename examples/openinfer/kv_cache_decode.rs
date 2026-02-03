use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/kv_cache_decode.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            q: f32[D];
        }

        constant {
            zero: f32[D];
        }

        volatile {
            k_hist: f32[4, D];
        }

        persistent {
            t: i32 @init(0);
            K(i): f32[D] @table @fixed(i=4);
        }

        block entry {
            op add(q, zero) >> q;
            cache.write q >> K[t];
            cache.increment 1 t;
            cache.read K[] >> k_hist;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let d = model.size_of("D")?;
    for step in 0..3 {
        let q = Random::<f32>::generate_with_seed(step as u64, (-1.0, 1.0), d)?;
        insert_executor!(exec, { q: q });
        exec.step()?;
        fetch_executor!(exec, { k_hist: Tensor<f32> });
        openinfer::log!("step {step} k_hist[0..4] = {:?}", &k_hist.data[..4.min(k_hist.len())]);
    }

    Ok(())
}
