use openinfer::{
    fetch_executor, format_truncated, graph, insert_executor, Device, ModelLoader, Random,
    Simulator, Tensor,
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

    let sim = Simulator::new(&model, &g, Device::Vulkan)?.with_timer();
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;
    insert_executor!(exec, { x: input });

    for mut node in exec.iterate() {
        let ev = node.event.clone();
        fetch_executor!(node, { y: Tensor<f32> });
        let y_str = format_truncated(&y.data);
        let y_pad = format!("{:<width$}", y_str, width = 32);
        println!(
            "y={} -- [{}] {} :: {} ({})",
            y_pad,
            ev.kind,
            ev.block_name,
            ev.op_name,
            ev.micros
        );
    }

    Ok(())
}
