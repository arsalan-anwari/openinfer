use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/linear_attention.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            Wq(head): f32[D, D] @pattern("attn.wq.{head}");
            Wk(head): f32[D, D] @pattern("attn.wk.{head}");
            Wv(head): f32[D, D] @pattern("attn.wv.{head}");
            w_out: f32[D, D];
        }

        volatile {
            q: f32[B, D];
            k: f32[B, D];
            h: f32[B, D];
            head_out: f32[B, D];
            out: f32[B, D];
        }

        block entry {
            op fill(out, value=0.0) >> out;
            loop heads (h_idx in 0..num_heads) {
                op matmul(x, Wq[h_idx]) >> q;
                op matmul(x, Wk[h_idx]) >> k;
                op add(q, k) >> h;
                op matmul(h, Wv[h_idx]) >> head_out;
                op add(out, head_out) >> out;
            }
            op matmul(out, w_out) >> out;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        2,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: x });
    exec.step()?;

    fetch_executor!(exec, { out: Tensor<f32> });
    openinfer::log!("out[0..8] = {:?}", &out.data[..8.min(out.len())]);

    Ok(())
}
