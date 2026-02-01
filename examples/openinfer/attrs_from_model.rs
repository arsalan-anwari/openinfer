use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../res/models/attrs_from_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            y_model: f32[B];
            y_hard: f32[B];
            cast_model: i32[B];
            cast_hard: i32[B];
        }

        constant {
            alpha: f32;
            clamp_max: f32;
        }

        block entry {
            op relu(x, alpha=alpha, clamp_max=clamp_max) >> y_model;
            op relu(x, alpha=0.2, clamp_max=2.5) >> y_hard;
            op cast(y_model, to=i32, rounding_mode=rounding_mode, saturate=true) >> cast_model;
            op cast(y_hard, to=i32, rounding_mode="nearest", saturate=false) >> cast_hard;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-3.0, 3.0), len)?;

    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(
        exec,
        { y_model: Tensor<f32>, y_hard: Tensor<f32>, cast_model: Tensor<i32>, cast_hard: Tensor<i32>, alpha: f32, clamp_max: f32 }
    );

    let rounding_mode = model
        .load_metadata_string("rounding_mode")?
        .unwrap_or_else(|| "<missing>".to_string());

    openinfer::log!(
        "model attrs: alpha={}, clamp_max={}, rounding_mode={}",
        alpha,
        clamp_max,
        rounding_mode
    );
    openinfer::log!("y_model[0..10] = {:?}", &y_model.data[..10.min(y_model.len())]);
    openinfer::log!("y_hard[0..10] = {:?}", &y_hard.data[..10.min(y_hard.len())]);
    openinfer::log!(
        "cast_model[0..10] = {:?}",
        &cast_model.data[..10.min(cast_model.len())]
    );
    openinfer::log!(
        "cast_hard[0..10] = {:?}",
        &cast_hard.data[..10.min(cast_hard.len())]
    );

    Ok(())
}
