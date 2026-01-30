use openinfer::{
    fetch_executor, graph, insert_executor,
    ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../res/models/yield_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            w: f32[D, D];
            bias: f32[D];
        }

        volatile {
            h: f32[B, D];
            h2: f32[B, D];
        }

        block entry {
            op matmul(x, w) >> h;
            yield x;

            op relu(h, alpha=0.0, clamp_max=6.0) >> h;

            await x;
            return;
        }

        block writer {
            await x;
            op add(x, bias) >> x;
            yield x;
        }

        block reader {
            await x;
            op relu(x, alpha=0.0, clamp_max=6.0) >> h2;
            yield x;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?
        .with_trace()
        .with_timer();
    
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        0,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;
    insert_executor!(exec, { x: x.clone() });

    exec.step()?;

    fetch_executor!(exec, { x: Tensor<f32> });
    openinfer::log!("x[0..8] = {:?}", &x.data[..8.min(x.len())]);

    Ok(())
}
