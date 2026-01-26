use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_table_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D];
        }

        volatile {
            out_full: f32[3, D];
            out_slice: f32[2, D];
            out_reset: f32[3, D];
        }

        persistent {
            A(i): f32[D] @table @fixed(i=3);
        }

        block entry {
            cache.write x >> A[0];
            cache.write x >> A[1];
            cache.write x >> A[2];

            cache.read A[] >> out_full;
            cache.read A[0..2] >> out_slice;

            cache.reset A[1];
            cache.read A[] >> out_reset;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("D")?;
    let input = Random::<f32>::generate_with_seed(1, (-1.0, 1.0), len)?;
    insert_executor!(exec, { x: input });
    exec.step()?;

    fetch_executor!(exec, { out_full: Tensor<f32>, out_slice: Tensor<f32>, out_reset: Tensor<f32> });
    openinfer::trace!("out_full shape = {:?}", out_full.shape());
    openinfer::trace!("out_slice shape = {:?}", out_slice.shape());
    openinfer::trace!(
        "out_reset[0..8] = {:?}",
        &out_reset.data[..8.min(out_reset.len())]
    );

    Ok(())
}
