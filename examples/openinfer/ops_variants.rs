use anyhow::Result;
use openinfer::{graph, Executor, ModelLoader, Simulator, Tensor, TensorElement, TensorOptions};
use std::path::Path;

mod util;
use util::select_device;

fn tensor_with_shape<T: TensorElement>(data: Vec<T>, shape: Vec<usize>) -> Result<Tensor<T>> {
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            ..TensorOptions::default()
        },
    )
}

fn insert_tensor<T: TensorElement>(
    exec: &mut Executor,
    name: &str,
    data: Vec<T>,
    shape: Vec<usize>,
) -> Result<()> {
    let tensor = tensor_with_shape(data, shape)?;
    exec.insert_dynamic(name, <T as TensorElement>::into_value(tensor))?;
    Ok(())
}

fn main() -> Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../res/models/ops_variants_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            a: f32[V];
            b: f32[V];
            b_b: f32[S];
            ai: i32[V];
            bi: i32[V];
            bi_b: i32[S];
            in_add: f32[V];
            in_add_b: f32[V];
            in_mul: f32[V];
            in_mul_b: f32[V];
            abs_in: f32[V];
            in_abs: f32[V];
            relu_in: f32[V];
            in_relu: f32[V];
            is_finite_in: f32[V];
            fill_in: f32[V];
            in_fill: f32[V];

            ma: f32[B, M, K];
            mb: f32[B, K, N];
            mb_b: f32[S, K, N];
            mai: i32[B, M, K];
            mbi: i32[B, K, N];
            mbi_b: i32[S, K, N];
            in_mm: f32[B, M, K];
            in_mm_b: f32[B, M, K];
        }

        volatile {
            add_out: f32[V];
            add_out_b: f32[V];
            add_acc: i64[V];
            add_acc_b: i64[V];
            mul_out: f32[V];
            mul_out_b: f32[V];
            mul_acc: i64[V];
            mul_acc_b: i64[V];
            mm_out: f32[B, M, N];
            mm_out_b: f32[B, M, N];
            mm_acc: i64[B, M, N];
            mm_acc_b: i64[B, M, N];
            abs_out: f32[V];
            relu_out: f32[V];
            finite_out: bool;
            fill_out: f32[V];
        }

        block entry {
            op add(a, b) >> add_out;
            op add(a, b_b) >> add_out_b;
            op add(in_add, b) >> in_add;
            op add(in_add_b, b_b) >> in_add_b;
            op add(ai, bi, acc=i64) >> add_acc;
            op add(ai, bi_b, acc=i64) >> add_acc_b;

            op mul(a, b) >> mul_out;
            op mul(a, b_b) >> mul_out_b;
            op mul(in_mul, b) >> in_mul;
            op mul(in_mul_b, b_b) >> in_mul_b;
            op mul(ai, bi, acc=i64) >> mul_acc;
            op mul(ai, bi_b, acc=i64) >> mul_acc_b;

            op matmul(ma, mb) >> mm_out;
            op matmul(ma, mb_b) >> mm_out_b;
            op matmul(in_mm, mb) >> in_mm;
            op matmul(in_mm_b, mb_b) >> in_mm_b;
            op matmul(mai, mbi, acc=i64) >> mm_acc;
            op matmul(mai, mbi_b, acc=i64) >> mm_acc_b;

            op abs(abs_in) >> abs_out;
            op abs(in_abs) >> in_abs;
            op relu(relu_in, alpha=0.1, clamp_max=2.0) >> relu_out;
            op relu(in_relu, alpha=0.1, clamp_max=2.0) >> in_relu;
            op is_finite(is_finite_in) >> finite_out;
            op fill(fill_in, value=1.0) >> fill_out;
            op fill(in_fill, value=2.0) >> in_fill;
            return;
        }
    };

    let v = model.size_of("V")?;
    let s = model.size_of("S")?;
    let b = model.size_of("B")?;
    let m = model.size_of("M")?;
    let k = model.size_of("K")?;
    let n = model.size_of("N")?;

    let a_vals: Vec<f32> = (0..v).map(|i| i as f32 * 0.1).collect();
    let b_vals: Vec<f32> = (0..v).map(|i| 1.0 + i as f32 * 0.05).collect();
    let b_b_vals: Vec<f32> = (0..s).map(|i| 2.0 + i as f32).collect();

    let ma_len = b * m * k;
    let mb_len = b * k * n;
    let mb_b_len = s * k * n;
    let ma_vals: Vec<f32> = (0..ma_len).map(|i| i as f32 * 0.01).collect();
    let mb_vals: Vec<f32> = (0..mb_len).map(|i| 1.0 + i as f32 * 0.02).collect();
    let mb_b_vals: Vec<f32> = (0..mb_b_len).map(|i| 3.0 + i as f32 * 0.03).collect();
    let ai_vals: Vec<i32> = (0..v).map(|i| i as i32 - 4).collect();
    let bi_vals: Vec<i32> = (0..v).map(|i| i as i32 - 2).collect();
    let bi_b_vals: Vec<i32> = (0..s).map(|i| i as i32).collect();
    let mai_vals: Vec<i32> = (0..ma_len).map(|i| i as i32 - 3).collect();
    let mbi_vals: Vec<i32> = (0..mb_len).map(|i| i as i32 + 1).collect();
    let mbi_b_vals: Vec<i32> = (0..mb_b_len).map(|i| i as i32 + 2).collect();

    let device = select_device()?;
    let sim = Simulator::new(&model, &g, device)?;
    let mut exec = sim.make_executor()?;

    insert_tensor(&mut exec, "a", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "b", b_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "b_b", b_b_vals.clone(), vec![s])?;
    insert_tensor(&mut exec, "ai", ai_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "bi", bi_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "bi_b", bi_b_vals.clone(), vec![s])?;
    insert_tensor(&mut exec, "in_add", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "in_add_b", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "in_mul", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "in_mul_b", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "abs_in", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "in_abs", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "relu_in", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "in_relu", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "is_finite_in", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "fill_in", a_vals.clone(), vec![v])?;
    insert_tensor(&mut exec, "in_fill", a_vals, vec![v])?;

    insert_tensor(&mut exec, "ma", ma_vals.clone(), vec![b, m, k])?;
    insert_tensor(&mut exec, "mb", mb_vals.clone(), vec![b, k, n])?;
    insert_tensor(&mut exec, "mb_b", mb_b_vals, vec![s, k, n])?;
    insert_tensor(&mut exec, "mai", mai_vals.clone(), vec![b, m, k])?;
    insert_tensor(&mut exec, "mbi", mbi_vals.clone(), vec![b, k, n])?;
    insert_tensor(&mut exec, "mbi_b", mbi_b_vals, vec![s, k, n])?;
    insert_tensor(&mut exec, "in_mm", ma_vals.clone(), vec![b, m, k])?;
    insert_tensor(&mut exec, "in_mm_b", ma_vals, vec![b, m, k])?;

    exec.step()?;
    openinfer::log!("ops_variants completed on {:?}", device);

    Ok(())
}
