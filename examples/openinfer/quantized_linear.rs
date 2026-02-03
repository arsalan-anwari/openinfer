use openinfer::{graph, ModelLoader, Simulator, TensorValue};
use std::path::Path;

mod util;
use util::select_device;

fn format_slice<T: std::fmt::Debug>(data: &[T]) -> String {
    let n = 8.min(data.len());
    format!("{:?}", &data[..n])
}

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/quantized_linear.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        constant {
            x: i4[B, D];
            w: i4[D, O];
        }

        volatile {
            y: i32[B, O];
        }

        block entry {
            op matmul(x, w, acc=i32) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;
    exec.step()?;

    let value: TensorValue = exec.fetch("y")?;
    match value {
        TensorValue::I32(t) => openinfer::log!("y: i32 {}", format_slice(&t.data)),
        other => openinfer::log!("y: dtype={:?}", other.dtype()),
    }

    Ok(())
}
