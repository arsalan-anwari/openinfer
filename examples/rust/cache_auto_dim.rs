use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Simulator, Tensor};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_auto_dim_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            bias: f32;
        }

        volatile {
            out_l: f32[D, H];
            out_m: f32[D, H];
            out_n: f32[D, H];
        }

        persistent {
            l_rows: i32 @init(0);
            l_cols: i32 @init(0);
            m_cols: i32 @init(0);
            n_rows: i32 @init(0);
            L(r, c): f32[D, H] @auto_dim(r, c);
            M(r, c): f32[D, H] @auto_dim(r, c);
            N(r, c): f32[D, H] @auto_dim(r, c);
        }

        block entry {
            cache.increment 3 l_rows;
            cache.increment 2 l_cols;
            cache.increment 5 m_cols;
            cache.increment 5 n_rows;

            cache.read L[l_rows, l_cols] >> out_l;
            cache.read M[0, m_cols] >> out_m;
            cache.read N[n_rows, 0] >> out_n;

            op add(out_l, bias) >> out_l;
            op add(out_m, bias) >> out_m;
            op add(out_n, bias) >> out_n;

            cache.write out_l >> L[l_rows, l_cols];
            cache.write out_m >> M[0, m_cols];
            cache.write out_n >> N[n_rows, 0];
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    insert_executor!(exec, { bias: 1.0f32 });
    exec.step()?;
    fetch_executor!(exec, { out_l: Tensor<f32>, out_m: Tensor<f32>, out_n: Tensor<f32> });
    openinfer::trace!("step 1 L shape = {:?}", out_l.shape());
    openinfer::trace!("step 1 M shape = {:?}", out_m.shape());
    openinfer::trace!("step 1 N shape = {:?}", out_n.shape());

    openinfer::trace!("");
    insert_executor!(exec, { bias: 1.0f32 });
    exec.step()?;
    fetch_executor!(exec, { out_l: Tensor<f32>, out_m: Tensor<f32>, out_n: Tensor<f32> });
    openinfer::trace!("step 2 L shape = {:?}", out_l.shape());
    openinfer::trace!("step 2 M shape = {:?}", out_m.shape());
    openinfer::trace!("step 2 N shape = {:?}", out_n.shape());

    openinfer::trace!("");
    insert_executor!(exec, { bias: 1.0f32 });
    exec.step()?;
    fetch_executor!(exec, { out_l: Tensor<f32>, out_m: Tensor<f32>, out_n: Tensor<f32> });
    openinfer::trace!("step 3 L shape = {:?}", out_l.shape());
    openinfer::trace!("step 3 M shape = {:?}", out_m.shape());
    openinfer::trace!("step 3 N shape = {:?}", out_n.shape());

    Ok(())
}
