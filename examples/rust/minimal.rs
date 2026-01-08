use openinfer::{
    graph, fetch_executor, insert_executor, Device, ModelLoader, Random, Simulator, Tensor,
};
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

    let sim = Simulator::new(&model, &g, Device::Cpu)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    println!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
