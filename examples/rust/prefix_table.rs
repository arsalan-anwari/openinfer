use openinfer::{
    fetch_executor, graph, insert_executor, Device, ModelLoader, Random, Simulator, Tensor,
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/prefix_table_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D];
        }

        volatile {
            y: f32[D];
            W(l): f32[D] @pattern("W.{l}");
        }

        constant {
            QKV(layer, head): f32[D] @pattern("QKV.{layer}.{head}");
        }

        block entry {
            op add(x, W[0]) >> y;
            op add(y, W[1]) >> y;
            op add(y, W[2]) >> y;
            op add(y, W[3]) >> y;
            op add(y, W[4]) >> y;
            op add(y, W[5]) >> y;
            op add(y, W[6]) >> y;
            op add(y, W[7]) >> y;
            op add(y, W[8]) >> y;
            op add(y, W[9]) >> y;
            op add(y, W[10]) >> y;
            op add(y, QKV[0, 0]) >> y;
            op add(y, QKV[1, 2]) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, Device::Cpu)?.with_trace();
    let mut exec = sim.make_executor()?;

    let len = model.size_of("D")?;
    let input = Random::<f32>::generate_with_seed(7, (-1.0, 1.0), len)?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    println!("y[0..8] = {:?}", &y.data[..8.min(y.len())]);

    Ok(())
}
