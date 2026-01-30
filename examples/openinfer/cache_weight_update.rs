use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_weight_update_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D];
            delta: f32[D];
        }

        volatile {
            tmp: f32[D];
            out: f32[D];
        }

        persistent {
            W: f32[D] @init(0.0);
        }

        block entry {
            cache.read W >> tmp;
            op add(tmp, x) >> tmp;
            op add(tmp, delta) >> tmp;
            cache.write tmp >> W;
            cache.read W >> out;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("D")?;
    let input = Random::<f32>::generate_with_seed(3, (-1.0, 1.0), len)?;
    let delta = Random::<f32>::generate_with_seed(9, (-0.1, 0.1), len)?;

    insert_executor!(exec, { x: input.clone(), delta: delta.clone() });
    exec.step()?;
    fetch_executor!(exec, { out: Tensor<f32> });
    openinfer::trace!("step 1 out[0..4] = {:?}", &out.data[..4.min(out.len())]);

    insert_executor!(exec, { x: input, delta: delta });
    exec.step()?;
    fetch_executor!(exec, { out: Tensor<f32> });
    openinfer::trace!("step 2 out[0..4] = {:?}", &out.data[..4.min(out.len())]);

    Ok(())
}
