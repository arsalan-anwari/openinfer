use openinfer::{
    graph, Bitset, Device, Executor, F16, ModelLoader, Simulator, Tensor, TensorElement,
    TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn tensor_with_shape<T: TensorElement>(
    data: Vec<T>,
    shape: Vec<usize>,
) -> anyhow::Result<Tensor<T>> {
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            ..TensorOptions::default()
        },
    )
}

fn compare_line<T: std::fmt::Debug + PartialEq>(label: &str, cpu: &[T], device: &[T]) {
    let n = cpu.len().min(device.len()).min(4);
    let cpu_preview = format!("{:?}", &cpu[..n]);
    let device_preview = format!("{:?}", &device[..n]);
    let ok = if cpu == device { "[✅]" } else { "[❌]" };
    log::info!("{ok} {label}[0..{n}] = {device_preview} -- CPUref: {cpu_preview}");
}

fn compare_row_line<T: std::fmt::Debug + PartialEq>(
    label: &str,
    cpu: &[T],
    device: &[T],
    row_len: usize,
) {
    let n = cpu.len().min(device.len()).min(row_len);
    let cpu_preview = format!("{:?}", &cpu[..n]);
    let device_preview = format!("{:?}", &device[..n]);
    let ok = if cpu == device { "[✅]" } else { "[❌]" };
    log::info!("{ok} {label}[0..{n}] = {device_preview} -- CPUref: {cpu_preview}");
}

fn compare_scalar(label: &str, cpu: bool, device: bool) {
    let ok = if cpu == device { "[✅]" } else { "[❌]" };
    log::info!("{ok} {label} = {device} -- CPUref: {cpu}");
}

fn compare_group<T: TensorElement + std::fmt::Debug + PartialEq>(
    cpu_exec: &mut Executor,
    device_exec: &mut Executor,
    tag: &str,
    row_len: usize,
    has_finite: bool,
    has_abs: bool,
    has_relu: bool,
) -> anyhow::Result<()> {
    let cpu_add: Tensor<T> = cpu_exec.fetch(&format!("add_{}", tag))?;
    let device_add: Tensor<T> = device_exec.fetch(&format!("add_{}", tag))?;
    let cpu_mul: Tensor<T> = cpu_exec.fetch(&format!("mul_{}", tag))?;
    let device_mul: Tensor<T> = device_exec.fetch(&format!("mul_{}", tag))?;
    let cpu_fill: Tensor<T> = cpu_exec.fetch(&format!("fill_{}", tag))?;
    let device_fill: Tensor<T> = device_exec.fetch(&format!("fill_{}", tag))?;
    let cpu_mm: Tensor<T> = cpu_exec.fetch(&format!("mm_{}", tag))?;
    let device_mm: Tensor<T> = device_exec.fetch(&format!("mm_{}", tag))?;

    log::info!("=== {} ===", tag);
    compare_line("add", &cpu_add.data, &device_add.data);
    compare_line("mul", &cpu_mul.data, &device_mul.data);
    if has_abs {
        let cpu_abs: Tensor<T> = cpu_exec.fetch(&format!("abs_{}", tag))?;
        let device_abs: Tensor<T> = device_exec.fetch(&format!("abs_{}", tag))?;
        compare_line("abs", &cpu_abs.data, &device_abs.data);
    }
    if has_relu {
        let cpu_relu: Tensor<T> = cpu_exec.fetch(&format!("relu_{}", tag))?;
        let device_relu: Tensor<T> = device_exec.fetch(&format!("relu_{}", tag))?;
        compare_line("relu", &cpu_relu.data, &device_relu.data);
    }
    compare_line("fill", &cpu_fill.data, &device_fill.data);
    if has_finite {
        let cpu_finite: bool = cpu_exec.fetch(&format!("finite_{}", tag))?;
        let device_finite: bool = device_exec.fetch(&format!("finite_{}", tag))?;
        compare_scalar("finite", cpu_finite, device_finite);
    }
    compare_row_line("mm", &cpu_mm.data, &device_mm.data, row_len);
    log::info!("");
    Ok(())
}

struct Inputs {
    a_i8: Tensor<i8>,
    b_i8: Tensor<i8>,
    ma_i8: Tensor<i8>,
    mb_i8: Tensor<i8>,
    a_i16: Tensor<i16>,
    b_i16: Tensor<i16>,
    ma_i16: Tensor<i16>,
    mb_i16: Tensor<i16>,
    a_i32: Tensor<i32>,
    b_i32: Tensor<i32>,
    ma_i32: Tensor<i32>,
    mb_i32: Tensor<i32>,
    a_i64: Tensor<i64>,
    b_i64: Tensor<i64>,
    ma_i64: Tensor<i64>,
    mb_i64: Tensor<i64>,
    a_u8: Tensor<u8>,
    b_u8: Tensor<u8>,
    ma_u8: Tensor<u8>,
    mb_u8: Tensor<u8>,
    a_u16: Tensor<u16>,
    b_u16: Tensor<u16>,
    ma_u16: Tensor<u16>,
    mb_u16: Tensor<u16>,
    a_u32: Tensor<u32>,
    b_u32: Tensor<u32>,
    ma_u32: Tensor<u32>,
    mb_u32: Tensor<u32>,
    a_u64: Tensor<u64>,
    b_u64: Tensor<u64>,
    ma_u64: Tensor<u64>,
    mb_u64: Tensor<u64>,
    a_f16: Tensor<F16>,
    b_f16: Tensor<F16>,
    ma_f16: Tensor<F16>,
    mb_f16: Tensor<F16>,
    a_f32: Tensor<f32>,
    b_f32: Tensor<f32>,
    ma_f32: Tensor<f32>,
    mb_f32: Tensor<f32>,
    a_f64: Tensor<f64>,
    b_f64: Tensor<f64>,
    ma_f64: Tensor<f64>,
    mb_f64: Tensor<f64>,
    a_bool: Tensor<bool>,
    b_bool: Tensor<bool>,
    ma_bool: Tensor<bool>,
    mb_bool: Tensor<bool>,
    a_bitset: Tensor<Bitset>,
    b_bitset: Tensor<Bitset>,
    ma_bitset: Tensor<Bitset>,
    mb_bitset: Tensor<Bitset>,
}

impl Inputs {
    fn insert_into(&self, exec: &mut Executor) -> anyhow::Result<()> {
        exec.insert_dynamic("a_i8", self.a_i8.clone())?;
        exec.insert_dynamic("b_i8", self.b_i8.clone())?;
        exec.insert_dynamic("ma_i8", self.ma_i8.clone())?;
        exec.insert_dynamic("mb_i8", self.mb_i8.clone())?;
        exec.insert_dynamic("a_i16", self.a_i16.clone())?;
        exec.insert_dynamic("b_i16", self.b_i16.clone())?;
        exec.insert_dynamic("ma_i16", self.ma_i16.clone())?;
        exec.insert_dynamic("mb_i16", self.mb_i16.clone())?;
        exec.insert_dynamic("a_i32", self.a_i32.clone())?;
        exec.insert_dynamic("b_i32", self.b_i32.clone())?;
        exec.insert_dynamic("ma_i32", self.ma_i32.clone())?;
        exec.insert_dynamic("mb_i32", self.mb_i32.clone())?;
        exec.insert_dynamic("a_i64", self.a_i64.clone())?;
        exec.insert_dynamic("b_i64", self.b_i64.clone())?;
        exec.insert_dynamic("ma_i64", self.ma_i64.clone())?;
        exec.insert_dynamic("mb_i64", self.mb_i64.clone())?;
        exec.insert_dynamic("a_u8", self.a_u8.clone())?;
        exec.insert_dynamic("b_u8", self.b_u8.clone())?;
        exec.insert_dynamic("ma_u8", self.ma_u8.clone())?;
        exec.insert_dynamic("mb_u8", self.mb_u8.clone())?;
        exec.insert_dynamic("a_u16", self.a_u16.clone())?;
        exec.insert_dynamic("b_u16", self.b_u16.clone())?;
        exec.insert_dynamic("ma_u16", self.ma_u16.clone())?;
        exec.insert_dynamic("mb_u16", self.mb_u16.clone())?;
        exec.insert_dynamic("a_u32", self.a_u32.clone())?;
        exec.insert_dynamic("b_u32", self.b_u32.clone())?;
        exec.insert_dynamic("ma_u32", self.ma_u32.clone())?;
        exec.insert_dynamic("mb_u32", self.mb_u32.clone())?;
        exec.insert_dynamic("a_u64", self.a_u64.clone())?;
        exec.insert_dynamic("b_u64", self.b_u64.clone())?;
        exec.insert_dynamic("ma_u64", self.ma_u64.clone())?;
        exec.insert_dynamic("mb_u64", self.mb_u64.clone())?;
        exec.insert_dynamic("a_f16", self.a_f16.clone())?;
        exec.insert_dynamic("b_f16", self.b_f16.clone())?;
        exec.insert_dynamic("ma_f16", self.ma_f16.clone())?;
        exec.insert_dynamic("mb_f16", self.mb_f16.clone())?;
        exec.insert_dynamic("a_f32", self.a_f32.clone())?;
        exec.insert_dynamic("b_f32", self.b_f32.clone())?;
        exec.insert_dynamic("ma_f32", self.ma_f32.clone())?;
        exec.insert_dynamic("mb_f32", self.mb_f32.clone())?;
        exec.insert_dynamic("a_f64", self.a_f64.clone())?;
        exec.insert_dynamic("b_f64", self.b_f64.clone())?;
        exec.insert_dynamic("ma_f64", self.ma_f64.clone())?;
        exec.insert_dynamic("mb_f64", self.mb_f64.clone())?;
        exec.insert_dynamic("a_bool", self.a_bool.clone())?;
        exec.insert_dynamic("b_bool", self.b_bool.clone())?;
        exec.insert_dynamic("ma_bool", self.ma_bool.clone())?;
        exec.insert_dynamic("mb_bool", self.mb_bool.clone())?;
        exec.insert_dynamic("a_bitset", self.a_bitset.clone())?;
        exec.insert_dynamic("b_bitset", self.b_bitset.clone())?;
        exec.insert_dynamic("ma_bitset", self.ma_bitset.clone())?;
        exec.insert_dynamic("mb_bitset", self.mb_bitset.clone())?;
        Ok(())
    }
}

fn build_inputs(v: usize, m: usize, k: usize, n: usize) -> anyhow::Result<Inputs> {
    let a_i8 = tensor_with_shape((0..v).map(|i| i as i8 - 4).collect(), vec![v])?;
    let b_i8 = tensor_with_shape((0..v).map(|i| i as i8 - 1).collect(), vec![v])?;
    let ma_i8 = tensor_with_shape((0..m * k).map(|i| i as i8 - 2).collect(), vec![m, k])?;
    let mb_i8 = tensor_with_shape((0..k * n).map(|i| i as i8 - 1).collect(), vec![k, n])?;

    let a_i16 = tensor_with_shape((0..v).map(|i| i as i16 - 4).collect(), vec![v])?;
    let b_i16 = tensor_with_shape((0..v).map(|i| i as i16 - 1).collect(), vec![v])?;
    let ma_i16 = tensor_with_shape((0..m * k).map(|i| i as i16 - 2).collect(), vec![m, k])?;
    let mb_i16 = tensor_with_shape((0..k * n).map(|i| i as i16 - 1).collect(), vec![k, n])?;

    let a_i32 = tensor_with_shape((0..v).map(|i| i as i32 - 4).collect(), vec![v])?;
    let b_i32 = tensor_with_shape((0..v).map(|i| i as i32 - 1).collect(), vec![v])?;
    let ma_i32 = tensor_with_shape((0..m * k).map(|i| i as i32 - 2).collect(), vec![m, k])?;
    let mb_i32 = tensor_with_shape((0..k * n).map(|i| i as i32 - 1).collect(), vec![k, n])?;

    let a_i64 = tensor_with_shape((0..v).map(|i| i as i64 - 4).collect(), vec![v])?;
    let b_i64 = tensor_with_shape((0..v).map(|i| i as i64 - 1).collect(), vec![v])?;
    let ma_i64 = tensor_with_shape((0..m * k).map(|i| i as i64 - 2).collect(), vec![m, k])?;
    let mb_i64 = tensor_with_shape((0..k * n).map(|i| i as i64 - 1).collect(), vec![k, n])?;

    let a_u8 = tensor_with_shape((0..v).map(|i| i as u8).collect(), vec![v])?;
    let b_u8 = tensor_with_shape((0..v).map(|i| (i as u8).wrapping_add(1)).collect(), vec![v])?;
    let ma_u8 = tensor_with_shape((0..m * k).map(|i| i as u8).collect(), vec![m, k])?;
    let mb_u8 = tensor_with_shape((0..k * n).map(|i| (i as u8).wrapping_add(1)).collect(), vec![k, n])?;

    let a_u16 = tensor_with_shape((0..v).map(|i| i as u16).collect(), vec![v])?;
    let b_u16 = tensor_with_shape((0..v).map(|i| (i as u16).wrapping_add(1)).collect(), vec![v])?;
    let ma_u16 = tensor_with_shape((0..m * k).map(|i| i as u16).collect(), vec![m, k])?;
    let mb_u16 = tensor_with_shape((0..k * n).map(|i| (i as u16).wrapping_add(1)).collect(), vec![k, n])?;

    let a_u32 = tensor_with_shape((0..v).map(|i| i as u32).collect(), vec![v])?;
    let b_u32 = tensor_with_shape((0..v).map(|i| (i as u32).wrapping_add(1)).collect(), vec![v])?;
    let ma_u32 = tensor_with_shape((0..m * k).map(|i| i as u32).collect(), vec![m, k])?;
    let mb_u32 = tensor_with_shape((0..k * n).map(|i| (i as u32).wrapping_add(1)).collect(), vec![k, n])?;

    let a_u64 = tensor_with_shape((0..v).map(|i| i as u64).collect(), vec![v])?;
    let b_u64 = tensor_with_shape((0..v).map(|i| (i as u64).wrapping_add(1)).collect(), vec![v])?;
    let ma_u64 = tensor_with_shape((0..m * k).map(|i| i as u64).collect(), vec![m, k])?;
    let mb_u64 = tensor_with_shape((0..k * n).map(|i| (i as u64).wrapping_add(1)).collect(), vec![k, n])?;

    let a_f16 = tensor_with_shape(
        (0..v)
            .map(|i| F16::from_f32(i as f32 * 0.5 - 2.0))
            .collect(),
        vec![v],
    )?;
    let b_f16 = tensor_with_shape(
        (0..v)
            .map(|i| F16::from_f32(i as f32 * 0.5 + 0.5))
            .collect(),
        vec![v],
    )?;
    let ma_f16 = tensor_with_shape(
        (0..m * k)
            .map(|i| F16::from_f32(i as f32 * 0.25 - 1.0))
            .collect(),
        vec![m, k],
    )?;
    let mb_f16 = tensor_with_shape(
        (0..k * n)
            .map(|i| F16::from_f32(i as f32 * 0.25 + 0.5))
            .collect(),
        vec![k, n],
    )?;

    let a_f32 = tensor_with_shape(
        (0..v).map(|i| i as f32 * 0.5 - 2.0).collect(),
        vec![v],
    )?;
    let b_f32 = tensor_with_shape(
        (0..v).map(|i| i as f32 * 0.5 + 0.5).collect(),
        vec![v],
    )?;
    let ma_f32 = tensor_with_shape(
        (0..m * k).map(|i| i as f32 * 0.25 - 1.0).collect(),
        vec![m, k],
    )?;
    let mb_f32 = tensor_with_shape(
        (0..k * n).map(|i| i as f32 * 0.25 + 0.5).collect(),
        vec![k, n],
    )?;

    let a_f64 = tensor_with_shape(
        (0..v).map(|i| i as f64 * 0.5 - 2.0).collect(),
        vec![v],
    )?;
    let b_f64 = tensor_with_shape(
        (0..v).map(|i| i as f64 * 0.5 + 0.5).collect(),
        vec![v],
    )?;
    let ma_f64 = tensor_with_shape(
        (0..m * k).map(|i| i as f64 * 0.25 - 1.0).collect(),
        vec![m, k],
    )?;
    let mb_f64 = tensor_with_shape(
        (0..k * n).map(|i| i as f64 * 0.25 + 0.5).collect(),
        vec![k, n],
    )?;

    let a_bool = tensor_with_shape((0..v).map(|i| i % 2 == 0).collect(), vec![v])?;
    let b_bool = tensor_with_shape((0..v).map(|i| i % 3 == 0).collect(), vec![v])?;
    let ma_bool = tensor_with_shape((0..m * k).map(|i| i % 2 == 0).collect(), vec![m, k])?;
    let mb_bool = tensor_with_shape((0..k * n).map(|i| i % 3 == 0).collect(), vec![k, n])?;

    let a_bitset = tensor_with_shape(
        (0..v)
            .map(|i| Bitset {
                bits: (i as u8).wrapping_mul(3),
            })
            .collect(),
        vec![v],
    )?;
    let b_bitset = tensor_with_shape(
        (0..v)
            .map(|i| Bitset {
                bits: (i as u8).wrapping_mul(5),
            })
            .collect(),
        vec![v],
    )?;
    let ma_bitset = tensor_with_shape(
        (0..m * k)
            .map(|i| Bitset {
                bits: (i as u8).wrapping_mul(2),
            })
            .collect(),
        vec![m, k],
    )?;
    let mb_bitset = tensor_with_shape(
        (0..k * n)
            .map(|i| Bitset {
                bits: (i as u8).wrapping_mul(7),
            })
            .collect(),
        vec![k, n],
    )?;

    Ok(Inputs {
        a_i8,
        b_i8,
        ma_i8,
        mb_i8,
        a_i16,
        b_i16,
        ma_i16,
        mb_i16,
        a_i32,
        b_i32,
        ma_i32,
        mb_i32,
        a_i64,
        b_i64,
        ma_i64,
        mb_i64,
        a_u8,
        b_u8,
        ma_u8,
        mb_u8,
        a_u16,
        b_u16,
        ma_u16,
        mb_u16,
        a_u32,
        b_u32,
        ma_u32,
        mb_u32,
        a_u64,
        b_u64,
        ma_u64,
        mb_u64,
        a_f16,
        b_f16,
        ma_f16,
        mb_f16,
        a_f32,
        b_f32,
        ma_f32,
        mb_f32,
        a_f64,
        b_f64,
        ma_f64,
        mb_f64,
        a_bool,
        b_bool,
        ma_bool,
        mb_bool,
        a_bitset,
        b_bitset,
        ma_bitset,
        mb_bitset,
    })
}

fn main() -> anyhow::Result<()> {
    let device = select_device()?;
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/ops_matrix_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            a_i8: i8[V];
            b_i8: i8[V];
            ma_i8: i8[M, K];
            mb_i8: i8[K, N];
            a_i16: i16[V];
            b_i16: i16[V];
            ma_i16: i16[M, K];
            mb_i16: i16[K, N];
            a_i32: i32[V];
            b_i32: i32[V];
            ma_i32: i32[M, K];
            mb_i32: i32[K, N];
            a_i64: i64[V];
            b_i64: i64[V];
            ma_i64: i64[M, K];
            mb_i64: i64[K, N];
            a_u8: u8[V];
            b_u8: u8[V];
            ma_u8: u8[M, K];
            mb_u8: u8[K, N];
            a_u16: u16[V];
            b_u16: u16[V];
            ma_u16: u16[M, K];
            mb_u16: u16[K, N];
            a_u32: u32[V];
            b_u32: u32[V];
            ma_u32: u32[M, K];
            mb_u32: u32[K, N];
            a_u64: u64[V];
            b_u64: u64[V];
            ma_u64: u64[M, K];
            mb_u64: u64[K, N];
            a_f16: f16[V];
            b_f16: f16[V];
            ma_f16: f16[M, K];
            mb_f16: f16[K, N];
            a_f32: f32[V];
            b_f32: f32[V];
            ma_f32: f32[M, K];
            mb_f32: f32[K, N];
            a_f64: f64[V];
            b_f64: f64[V];
            ma_f64: f64[M, K];
            mb_f64: f64[K, N];
            a_bool: bool[V];
            b_bool: bool[V];
            ma_bool: bool[M, K];
            mb_bool: bool[K, N];
            a_bitset: bitset[V];
            b_bitset: bitset[V];
            ma_bitset: bitset[M, K];
            mb_bitset: bitset[K, N];
        }

        volatile {
            add_i8: i8[V];
            mul_i8: i8[V];
            abs_i8: i8[V];
            relu_i8: i8[V];
            fill_i8: i8[V];
            mm_i8: i8[M, N];
            add_i16: i16[V];
            mul_i16: i16[V];
            abs_i16: i16[V];
            relu_i16: i16[V];
            fill_i16: i16[V];
            mm_i16: i16[M, N];
            add_i32: i32[V];
            mul_i32: i32[V];
            abs_i32: i32[V];
            relu_i32: i32[V];
            fill_i32: i32[V];
            mm_i32: i32[M, N];
            add_i64: i64[V];
            mul_i64: i64[V];
            abs_i64: i64[V];
            relu_i64: i64[V];
            fill_i64: i64[V];
            mm_i64: i64[M, N];
            add_u8: u8[V];
            mul_u8: u8[V];
            fill_u8: u8[V];
            mm_u8: u8[M, N];
            add_u16: u16[V];
            mul_u16: u16[V];
            fill_u16: u16[V];
            mm_u16: u16[M, N];
            add_u32: u32[V];
            mul_u32: u32[V];
            fill_u32: u32[V];
            mm_u32: u32[M, N];
            add_u64: u64[V];
            mul_u64: u64[V];
            fill_u64: u64[V];
            mm_u64: u64[M, N];
            add_f16: f16[V];
            mul_f16: f16[V];
            abs_f16: f16[V];
            relu_f16: f16[V];
            fill_f16: f16[V];
            finite_f16: bool;
            mm_f16: f16[M, N];
            add_f32: f32[V];
            mul_f32: f32[V];
            abs_f32: f32[V];
            relu_f32: f32[V];
            fill_f32: f32[V];
            finite_f32: bool;
            mm_f32: f32[M, N];
            add_f64: f64[V];
            mul_f64: f64[V];
            abs_f64: f64[V];
            relu_f64: f64[V];
            fill_f64: f64[V];
            finite_f64: bool;
            mm_f64: f64[M, N];
            add_bool: bool[V];
            mul_bool: bool[V];
            fill_bool: bool[V];
            mm_bool: bool[M, N];
            add_bitset: bitset[V];
            mul_bitset: bitset[V];
            fill_bitset: bitset[V];
            mm_bitset: bitset[M, N];
        }

        block entry {
            op add(a_i8, b_i8) >> add_i8;
            op mul(a_i8, b_i8) >> mul_i8;
            op abs(a_i8) >> abs_i8;
            op relu(a_i8, alpha=1, clamp_max=2) >> relu_i8;
            op fill(a_i8, value=1) >> fill_i8;
            op matmul(ma_i8, mb_i8) >> mm_i8;

            op add(a_i16, b_i16) >> add_i16;
            op mul(a_i16, b_i16) >> mul_i16;
            op abs(a_i16) >> abs_i16;
            op relu(a_i16, alpha=1, clamp_max=2) >> relu_i16;
            op fill(a_i16, value=1) >> fill_i16;
            op matmul(ma_i16, mb_i16) >> mm_i16;

            op add(a_i32, b_i32) >> add_i32;
            op mul(a_i32, b_i32) >> mul_i32;
            op abs(a_i32) >> abs_i32;
            op relu(a_i32, alpha=1, clamp_max=2) >> relu_i32;
            op fill(a_i32, value=1) >> fill_i32;
            op matmul(ma_i32, mb_i32) >> mm_i32;

            op add(a_i64, b_i64) >> add_i64;
            op mul(a_i64, b_i64) >> mul_i64;
            op abs(a_i64) >> abs_i64;
            op relu(a_i64, alpha=1, clamp_max=2) >> relu_i64;
            op fill(a_i64, value=1) >> fill_i64;
            op matmul(ma_i64, mb_i64) >> mm_i64;

            op add(a_u8, b_u8) >> add_u8;
            op mul(a_u8, b_u8) >> mul_u8;
            op fill(a_u8, value=1u8) >> fill_u8;
            op matmul(ma_u8, mb_u8) >> mm_u8;

            op add(a_u16, b_u16) >> add_u16;
            op mul(a_u16, b_u16) >> mul_u16;
            op fill(a_u16, value=1u16) >> fill_u16;
            op matmul(ma_u16, mb_u16) >> mm_u16;

            op add(a_u32, b_u32) >> add_u32;
            op mul(a_u32, b_u32) >> mul_u32;
            op fill(a_u32, value=1u32) >> fill_u32;
            op matmul(ma_u32, mb_u32) >> mm_u32;

            op add(a_u64, b_u64) >> add_u64;
            op mul(a_u64, b_u64) >> mul_u64;
            op fill(a_u64, value=1u64) >> fill_u64;
            op matmul(ma_u64, mb_u64) >> mm_u64;

            op add(a_f16, b_f16) >> add_f16;
            op mul(a_f16, b_f16) >> mul_f16;
            op abs(a_f16) >> abs_f16;
            op relu(a_f16, alpha=0.1, clamp_max=2.0) >> relu_f16;
            op fill(a_f16, value=1.0) >> fill_f16;
            op is_finite(a_f16) >> finite_f16;
            op matmul(ma_f16, mb_f16) >> mm_f16;

            op add(a_f32, b_f32) >> add_f32;
            op mul(a_f32, b_f32) >> mul_f32;
            op abs(a_f32) >> abs_f32;
            op relu(a_f32, alpha=0.1, clamp_max=2.0) >> relu_f32;
            op fill(a_f32, value=1.0) >> fill_f32;
            op is_finite(a_f32) >> finite_f32;
            op matmul(ma_f32, mb_f32) >> mm_f32;

            op add(a_f64, b_f64) >> add_f64;
            op mul(a_f64, b_f64) >> mul_f64;
            op abs(a_f64) >> abs_f64;
            op relu(a_f64, alpha=0.1, clamp_max=2.0) >> relu_f64;
            op fill(a_f64, value=1.0) >> fill_f64;
            op is_finite(a_f64) >> finite_f64;
            op matmul(ma_f64, mb_f64) >> mm_f64;

            op add(a_bool, b_bool) >> add_bool;
            op mul(a_bool, b_bool) >> mul_bool;
            op fill(a_bool, value=true) >> fill_bool;
            op matmul(ma_bool, mb_bool) >> mm_bool;

            op add(a_bitset, b_bitset) >> add_bitset;
            op mul(a_bitset, b_bitset) >> mul_bitset;
            op fill(a_bitset, value=255u8) >> fill_bitset;
            op matmul(ma_bitset, mb_bitset) >> mm_bitset;
            return;
        }
    };

    let v = model.size_of("V")?;
    let m = model.size_of("M")?;
    let k = model.size_of("K")?;
    let n = model.size_of("N")?;

    let inputs = build_inputs(v, m, k, n)?;

    let cpu_sim = Simulator::new(&model, &g, Device::Cpu)?;
    let mut cpu_exec = cpu_sim.make_executor()?;
    inputs.insert_into(&mut cpu_exec)?;
    cpu_exec.step()?;

    let sim = Simulator::new(&model, &g, device)?;
    let mut exec = sim.make_executor()?;
    inputs.insert_into(&mut exec)?;
    exec.step()?;

    compare_group::<i8>(&mut cpu_exec, &mut exec, "i8", n, false, true, true)?;
    compare_group::<i16>(&mut cpu_exec, &mut exec, "i16", n, false, true, true)?;
    compare_group::<i32>(&mut cpu_exec, &mut exec, "i32", n, false, true, true)?;
    compare_group::<i64>(&mut cpu_exec, &mut exec, "i64", n, false, true, true)?;
    compare_group::<u8>(&mut cpu_exec, &mut exec, "u8", n, false, false, false)?;
    compare_group::<u16>(&mut cpu_exec, &mut exec, "u16", n, false, false, false)?;
    compare_group::<u32>(&mut cpu_exec, &mut exec, "u32", n, false, false, false)?;
    compare_group::<u64>(&mut cpu_exec, &mut exec, "u64", n, false, false, false)?;
    compare_group::<F16>(&mut cpu_exec, &mut exec, "f16", n, true, true, true)?;
    compare_group::<f32>(&mut cpu_exec, &mut exec, "f32", n, true, true, true)?;
    compare_group::<f64>(&mut cpu_exec, &mut exec, "f64", n, true, true, true)?;
    compare_group::<bool>(&mut cpu_exec, &mut exec, "bool", n, false, false, false)?;
    compare_group::<Bitset>(&mut cpu_exec, &mut exec, "bitset", n, false, false, false)?;

    log::info!("ops_matrix completed");

    Ok(())
}
