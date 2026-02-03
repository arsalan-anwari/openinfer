use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_window_slice.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D];
        }

        constant {
            limit: i32;
            zero: i32;
        }

        volatile {
            window: f32[3, D];
            start_val: i32;
            end_val: i32;
            start_clamped: i32;
            end_clamped: i32;
        }

        persistent {
            write_idx: i32 @init(0);
            start: i32 @init(0);
            end: i32 @init(3);
            A(i): f32[D] @table @fixed(i=6);
        }

        block entry {
            cache.write x >> A[write_idx];
            cache.increment 1 write_idx;

            cache.read start >> start_val;
            cache.read end >> end_val;
            op max(start_val, zero) >> start_clamped;
            op min(end_val, limit) >> end_clamped;
            cache.write start_clamped >> start;
            cache.write end_clamped >> end;
            cache.read A[start..end] >> window;

            cache.increment 1 start;
            cache.increment 1 end;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let d = model.size_of("D")?;
    for step in 0..4 {
        let x = Random::<f32>::generate_with_seed(step as u64, (-1.0, 1.0), d)?;
        insert_executor!(exec, { x: x });
        exec.step()?;
        fetch_executor!(exec, { window: Tensor<f32> });
        openinfer::log!(
            "step {step} window[0..4] = {:?}",
            &window.data[..4.min(window.len())]
        );
    }

    Ok(())
}
