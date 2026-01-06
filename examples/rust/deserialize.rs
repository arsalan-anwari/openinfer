use openinfer::{fetch_executor, insert_executor, Device, GraphDeserialize, ModelLoader, Simulator};
use rand::Rng;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/minimal_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let graph_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/rust/out/minimal-graph.json");
    let graph_txt = std::fs::read_to_string(graph_path)?;
    let graph_json = serde_json::from_str(&graph_txt)?;
    let g = GraphDeserialize::from_json(graph_json)?;

    let sim = Simulator::new(&model, Device::Cpu)?;
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

    fetch_executor!(exec, { y: f32 });
    println!("y[0..100] = {:?}", &y.data[..100.min(y.len())]);

    Ok(())
}
