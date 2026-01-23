use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/reference_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            state: f32[B] @ref("state.0");
            out: f32[B];
        }

        constant {
            weight: f32[B] @ref("weight.0");
            bias: f32 @ref("bias.0");
        }

        block entry {
            assign t0: f32[B];
            op add(x, weight) >> t0;
            op add(t0, state) >> out;
            op add(out, bias) >> out;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(7, (-2.0, 2.0), len)?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { out: Tensor<f32> });
    println!("out[0..8] = {:?}", &out.data[..8.min(out.len())]);

    Ok(())
}
