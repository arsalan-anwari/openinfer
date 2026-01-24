use openinfer::{
    fetch_executor, insert_executor, GraphDeserialize, ModelLoader, Random, Simulator,
    Tensor,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let graph_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/rust/out/minimal-graph.json");
    let graph_txt = std::fs::read_to_string(graph_path)?;
    let graph_json = serde_json::from_str(&graph_txt)?;
    let g = GraphDeserialize::from_json(graph_json)?;

    let sim = Simulator::new(&model, &g, select_device()?)?.with_trace();
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;
    insert_executor!(exec, { x: input });

    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    log::info!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
