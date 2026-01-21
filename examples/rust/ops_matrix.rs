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

fn print_line<T: std::fmt::Debug>(label: &str, data: &[T], cpu_ref: &str) {
    let n = 4.min(data.len());
    let actual = format!("{:?}", &data[..n]);
    let ok = if actual == cpu_ref { "[✅]" } else { "[❌]" };
    println!("{ok} {label}[0..{n}] = {actual} -- CPUref: {cpu_ref}");
}

fn print_row_line<T: std::fmt::Debug>(label: &str, data: &[T], row_len: usize, cpu_ref: &str) {
    let n = row_len.min(data.len());
    let actual = format!("{:?}", &data[..n]);
    let ok = if actual == cpu_ref { "[✅]" } else { "[❌]" };
    println!("{ok} {label}[0..{n}] = {actual} -- CPUref: {cpu_ref}");
}

struct CpuRefs {
    add: &'static str,
    mul: &'static str,
    abs: &'static str,
    relu: &'static str,
    fill: &'static str,
    finite: &'static str,
    mm: &'static str,
}

fn cpu_refs(tag: &str) -> CpuRefs {
    match tag {
        "i8" => CpuRefs {
            add: "[-5, -3, -1, 1]",
            mul: "[4, 0, -2, -2]",
            abs: "[4, 3, 2, 1]",
            relu: "[0, 0, 0, 0]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[-1, -4, -7, -10]",
        },
        "i16" => CpuRefs {
            add: "[-5, -3, -1, 1]",
            mul: "[4, 0, -2, -2]",
            abs: "[4, 3, 2, 1]",
            relu: "[0, 0, 0, 0]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[-1, -4, -7, -10]",
        },
        "i32" => CpuRefs {
            add: "[-5, -3, -1, 1]",
            mul: "[4, 0, -2, -2]",
            abs: "[4, 3, 2, 1]",
            relu: "[0, 0, 0, 0]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[-1, -4, -7, -10]",
        },
        "i64" => CpuRefs {
            add: "[-5, -3, -1, 1]",
            mul: "[4, 0, -2, -2]",
            abs: "[4, 3, 2, 1]",
            relu: "[0, 0, 0, 0]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[-1, -4, -7, -10]",
        },
        "u8" => CpuRefs {
            add: "[1, 3, 5, 7]",
            mul: "[0, 2, 6, 12]",
            abs: "[0, 1, 2, 3]",
            relu: "[0, 1, 2, 2]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[23, 26, 29, 32]",
        },
        "u16" => CpuRefs {
            add: "[1, 3, 5, 7]",
            mul: "[0, 2, 6, 12]",
            abs: "[0, 1, 2, 3]",
            relu: "[0, 1, 2, 2]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[23, 26, 29, 32]",
        },
        "u32" => CpuRefs {
            add: "[1, 3, 5, 7]",
            mul: "[0, 2, 6, 12]",
            abs: "[0, 1, 2, 3]",
            relu: "[0, 1, 2, 2]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[23, 26, 29, 32]",
        },
        "u64" => CpuRefs {
            add: "[1, 3, 5, 7]",
            mul: "[0, 2, 6, 12]",
            abs: "[0, 1, 2, 3]",
            relu: "[0, 1, 2, 2]",
            fill: "[1, 1, 1, 1]",
            finite: "true",
            mm: "[23, 26, 29, 32]",
        },
        "f16" => CpuRefs {
            add: "[F16 { bits: 48640 }, F16 { bits: 47104 }, F16 { bits: 14336 }, F16 { bits: 15872 }]",
            mul: "[F16 { bits: 48128 }, F16 { bits: 48640 }, F16 { bits: 48640 }, F16 { bits: 48128 }]",
            abs: "[F16 { bits: 16384 }, F16 { bits: 15872 }, F16 { bits: 15360 }, F16 { bits: 14336 }]",
            relu: "[F16 { bits: 45670 }, F16 { bits: 45261 }, F16 { bits: 44646 }, F16 { bits: 43622 }]",
            fill: "[F16 { bits: 15360 }, F16 { bits: 15360 }, F16 { bits: 15360 }, F16 { bits: 15360 }]",
            finite: "true",
            mm: "[F16 { bits: 49600 }, F16 { bits: 49888 }, F16 { bits: 50176 }, F16 { bits: 50320 }]",
        },
        "f32" => CpuRefs {
            add: "[-1.5, -0.5, 0.5, 1.5]",
            mul: "[-1.0, -1.5, -1.5, -1.0]",
            abs: "[2.0, 1.5, 1.0, 0.5]",
            relu: "[-0.2, -0.15, -0.1, -0.05]",
            fill: "[1.0, 1.0, 1.0, 1.0]",
            finite: "true",
            mm: "[-2.875, -3.4375, -4.0, -4.5625]",
        },
        "f64" => CpuRefs {
            add: "[-1.5, -0.5, 0.5, 1.5]",
            mul: "[-1.0, -1.5, -1.5, -1.0]",
            abs: "[2.0, 1.5, 1.0, 0.5]",
            relu: "[-0.20000000298023224, -0.15000000223517418, -0.10000000149011612, -0.05000000074505806]",
            fill: "[1.0, 1.0, 1.0, 1.0]",
            finite: "true",
            mm: "[-2.875, -3.4375, -4.0, -4.5625]",
        },
        "bool" => CpuRefs {
            add: "[true, false, true, true]",
            mul: "[true, false, false, false]",
            abs: "[true, false, true, false]",
            relu: "[true, false, true, false]",
            fill: "[true, true, true, true]",
            finite: "true",
            mm: "[true, true, false, true]",
        },
        "bitset" => CpuRefs {
            add: "[Bitset { bits: 0 }, Bitset { bits: 8 }, Bitset { bits: 16 }, Bitset { bits: 24 }]",
            mul: "[Bitset { bits: 0 }, Bitset { bits: 15 }, Bitset { bits: 60 }, Bitset { bits: 135 }]",
            abs: "[Bitset { bits: 0 }, Bitset { bits: 3 }, Bitset { bits: 6 }, Bitset { bits: 9 }]",
            relu: "[Bitset { bits: 0 }, Bitset { bits: 2 }, Bitset { bits: 2 }, Bitset { bits: 2 }]",
            fill: "[Bitset { bits: 255 }, Bitset { bits: 255 }, Bitset { bits: 255 }, Bitset { bits: 255 }]",
            finite: "true",
            mm: "[Bitset { bits: 24 }, Bitset { bits: 66 }, Bitset { bits: 108 }, Bitset { bits: 150 }]",
        },
        _ => CpuRefs {
            add: "[]",
            mul: "[]",
            abs: "[]",
            relu: "[]",
            fill: "[]",
            finite: "false",
            mm: "[]",
        },
    }
}

fn print_group<T: TensorElement + std::fmt::Debug>(
    exec: &mut Executor,
    tag: &str,
    row_len: usize,
    has_finite: bool,
    has_abs: bool,
    has_relu: bool,
) -> anyhow::Result<()> {
    let refs = cpu_refs(tag);
    let add: Tensor<T> = exec.fetch(&format!("add_{}", tag))?;
    let mul: Tensor<T> = exec.fetch(&format!("mul_{}", tag))?;
    let fill: Tensor<T> = exec.fetch(&format!("fill_{}", tag))?;
    let mm: Tensor<T> = exec.fetch(&format!("mm_{}", tag))?;

    println!("=== {} ===", tag);
    print_line("add", &add.data, refs.add);
    print_line("mul", &mul.data, refs.mul);
    if has_abs {
        let abs: Tensor<T> = exec.fetch(&format!("abs_{}", tag))?;
        print_line("abs", &abs.data, refs.abs);
    }
    if has_relu {
        let relu: Tensor<T> = exec.fetch(&format!("relu_{}", tag))?;
        print_line("relu", &relu.data, refs.relu);
    }
    print_line("fill", &fill.data, refs.fill);
    if has_finite {
        let finite: bool = exec.fetch(&format!("finite_{}", tag))?;
        let actual = format!("{}", finite);
        let ok = if actual == refs.finite { "[✅]" } else { "[❌]" };
        println!("{ok} finite = {actual} -- CPUref: {}", refs.finite);
    }
    print_row_line("mm", &mm.data, row_len, refs.mm);
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = select_device()?;
    let is_vulkan = matches!(device, Device::Vulkan);
    let model_path = if is_vulkan {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/ops_matrix_model_vulkan.oinf")
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/ops_matrix_model.oinf")
    };
    let model = ModelLoader::open(model_path)?;

    let g = if is_vulkan {
        graph! {
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
                a_f32: f32[V];
                b_f32: f32[V];
                ma_f32: f32[M, K];
                mb_f32: f32[K, N];
                a_bool: bool[V];
                b_bool: bool[V];
                ma_bool: bool[M, K];
                mb_bool: bool[K, N];
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
                add_f32: f32[V];
                mul_f32: f32[V];
                abs_f32: f32[V];
                relu_f32: f32[V];
                fill_f32: f32[V];
                finite_f32: bool;
                mm_f32: f32[M, N];
                add_bool: bool[V];
                mul_bool: bool[V];
                fill_bool: bool[V];
                mm_bool: bool[M, N];
            }

            block entry {
                op add(a_i8, b_i8) >> add_i8;
                op mul(a_i8, b_i8) >> mul_i8;
                op abs(a_i8) >> abs_i8;
                op relu(a_i8, negative_slope=0.1, clamp_max=2.0) >> relu_i8;
                op fill(a_i8, value=1) >> fill_i8;
                op matmul(ma_i8, mb_i8) >> mm_i8;

                op add(a_i16, b_i16) >> add_i16;
                op mul(a_i16, b_i16) >> mul_i16;
                op abs(a_i16) >> abs_i16;
                op relu(a_i16, negative_slope=0.1, clamp_max=2.0) >> relu_i16;
                op fill(a_i16, value=1) >> fill_i16;
                op matmul(ma_i16, mb_i16) >> mm_i16;

                op add(a_i32, b_i32) >> add_i32;
                op mul(a_i32, b_i32) >> mul_i32;
                op abs(a_i32) >> abs_i32;
                op relu(a_i32, negative_slope=0.1, clamp_max=2.0) >> relu_i32;
                op fill(a_i32, value=1) >> fill_i32;
                op matmul(ma_i32, mb_i32) >> mm_i32;

                op add(a_i64, b_i64) >> add_i64;
                op mul(a_i64, b_i64) >> mul_i64;
                op abs(a_i64) >> abs_i64;
                op relu(a_i64, negative_slope=0.1, clamp_max=2.0) >> relu_i64;
                op fill(a_i64, value=1) >> fill_i64;
                op matmul(ma_i64, mb_i64) >> mm_i64;

                op add(a_u8, b_u8) >> add_u8;
                op mul(a_u8, b_u8) >> mul_u8;
                op fill(a_u8, value=1) >> fill_u8;
                op matmul(ma_u8, mb_u8) >> mm_u8;

                op add(a_u16, b_u16) >> add_u16;
                op mul(a_u16, b_u16) >> mul_u16;
                op fill(a_u16, value=1) >> fill_u16;
                op matmul(ma_u16, mb_u16) >> mm_u16;

                op add(a_u32, b_u32) >> add_u32;
                op mul(a_u32, b_u32) >> mul_u32;
                op fill(a_u32, value=1) >> fill_u32;
                op matmul(ma_u32, mb_u32) >> mm_u32;

                op add(a_u64, b_u64) >> add_u64;
                op mul(a_u64, b_u64) >> mul_u64;
                op fill(a_u64, value=1) >> fill_u64;
                op matmul(ma_u64, mb_u64) >> mm_u64;

                op add(a_f32, b_f32) >> add_f32;
                op mul(a_f32, b_f32) >> mul_f32;
                op abs(a_f32) >> abs_f32;
                op relu(a_f32, negative_slope=0.1, clamp_max=2.0) >> relu_f32;
                op fill(a_f32, value=1.0) >> fill_f32;
                op is_finite(a_f32) >> finite_f32;
                op matmul(ma_f32, mb_f32) >> mm_f32;

                op add(a_bool, b_bool) >> add_bool;
                op mul(a_bool, b_bool) >> mul_bool;
                op fill(a_bool, value=true) >> fill_bool;
                op matmul(ma_bool, mb_bool) >> mm_bool;
                return;
            }
        }
    } else {
        graph! {
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
                op relu(a_i8, negative_slope=0.1, clamp_max=2.0) >> relu_i8;
                op fill(a_i8, value=1) >> fill_i8;
                op matmul(ma_i8, mb_i8) >> mm_i8;

                op add(a_i16, b_i16) >> add_i16;
                op mul(a_i16, b_i16) >> mul_i16;
                op abs(a_i16) >> abs_i16;
                op relu(a_i16, negative_slope=0.1, clamp_max=2.0) >> relu_i16;
                op fill(a_i16, value=1) >> fill_i16;
                op matmul(ma_i16, mb_i16) >> mm_i16;

                op add(a_i32, b_i32) >> add_i32;
                op mul(a_i32, b_i32) >> mul_i32;
                op abs(a_i32) >> abs_i32;
                op relu(a_i32, negative_slope=0.1, clamp_max=2.0) >> relu_i32;
                op fill(a_i32, value=1) >> fill_i32;
                op matmul(ma_i32, mb_i32) >> mm_i32;

                op add(a_i64, b_i64) >> add_i64;
                op mul(a_i64, b_i64) >> mul_i64;
                op abs(a_i64) >> abs_i64;
                op relu(a_i64, negative_slope=0.1, clamp_max=2.0) >> relu_i64;
                op fill(a_i64, value=1) >> fill_i64;
                op matmul(ma_i64, mb_i64) >> mm_i64;

                op add(a_u8, b_u8) >> add_u8;
                op mul(a_u8, b_u8) >> mul_u8;
                op fill(a_u8, value=1) >> fill_u8;
                op matmul(ma_u8, mb_u8) >> mm_u8;

                op add(a_u16, b_u16) >> add_u16;
                op mul(a_u16, b_u16) >> mul_u16;
                op fill(a_u16, value=1) >> fill_u16;
                op matmul(ma_u16, mb_u16) >> mm_u16;

                op add(a_u32, b_u32) >> add_u32;
                op mul(a_u32, b_u32) >> mul_u32;
                op fill(a_u32, value=1) >> fill_u32;
                op matmul(ma_u32, mb_u32) >> mm_u32;

                op add(a_u64, b_u64) >> add_u64;
                op mul(a_u64, b_u64) >> mul_u64;
                op fill(a_u64, value=1) >> fill_u64;
                op matmul(ma_u64, mb_u64) >> mm_u64;

                op add(a_f16, b_f16) >> add_f16;
                op mul(a_f16, b_f16) >> mul_f16;
                op abs(a_f16) >> abs_f16;
                op relu(a_f16, negative_slope=0.1, clamp_max=2.0) >> relu_f16;
                op fill(a_f16, value=1.0) >> fill_f16;
                op is_finite(a_f16) >> finite_f16;
                op matmul(ma_f16, mb_f16) >> mm_f16;

                op add(a_f32, b_f32) >> add_f32;
                op mul(a_f32, b_f32) >> mul_f32;
                op abs(a_f32) >> abs_f32;
                op relu(a_f32, negative_slope=0.1, clamp_max=2.0) >> relu_f32;
                op fill(a_f32, value=1.0) >> fill_f32;
                op is_finite(a_f32) >> finite_f32;
                op matmul(ma_f32, mb_f32) >> mm_f32;

                op add(a_f64, b_f64) >> add_f64;
                op mul(a_f64, b_f64) >> mul_f64;
                op abs(a_f64) >> abs_f64;
                op relu(a_f64, negative_slope=0.1, clamp_max=2.0) >> relu_f64;
                op fill(a_f64, value=1.0) >> fill_f64;
                op is_finite(a_f64) >> finite_f64;
                op matmul(ma_f64, mb_f64) >> mm_f64;

                op add(a_bool, b_bool) >> add_bool;
                op mul(a_bool, b_bool) >> mul_bool;
                op fill(a_bool, value=true) >> fill_bool;
                op matmul(ma_bool, mb_bool) >> mm_bool;

                op add(a_bitset, b_bitset) >> add_bitset;
                op mul(a_bitset, b_bitset) >> mul_bitset;
                op fill(a_bitset, value=1) >> fill_bitset;
                op matmul(ma_bitset, mb_bitset) >> mm_bitset;
                return;
            }
        }
    };

    let sim = Simulator::new(&model, &g, device)?;
    let mut exec = sim.make_executor()?;

    let v = model.size_of("V")?;
    let m = model.size_of("M")?;
    let k = model.size_of("K")?;
    let n = model.size_of("N")?;

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

    exec.insert_dynamic("a_i8", a_i8)?;
    exec.insert_dynamic("b_i8", b_i8)?;
    exec.insert_dynamic("ma_i8", ma_i8)?;
    exec.insert_dynamic("mb_i8", mb_i8)?;
    exec.insert_dynamic("a_i16", a_i16)?;
    exec.insert_dynamic("b_i16", b_i16)?;
    exec.insert_dynamic("ma_i16", ma_i16)?;
    exec.insert_dynamic("mb_i16", mb_i16)?;
    exec.insert_dynamic("a_i32", a_i32)?;
    exec.insert_dynamic("b_i32", b_i32)?;
    exec.insert_dynamic("ma_i32", ma_i32)?;
    exec.insert_dynamic("mb_i32", mb_i32)?;
    exec.insert_dynamic("a_i64", a_i64)?;
    exec.insert_dynamic("b_i64", b_i64)?;
    exec.insert_dynamic("ma_i64", ma_i64)?;
    exec.insert_dynamic("mb_i64", mb_i64)?;
    exec.insert_dynamic("a_u8", a_u8)?;
    exec.insert_dynamic("b_u8", b_u8)?;
    exec.insert_dynamic("ma_u8", ma_u8)?;
    exec.insert_dynamic("mb_u8", mb_u8)?;
    exec.insert_dynamic("a_u16", a_u16)?;
    exec.insert_dynamic("b_u16", b_u16)?;
    exec.insert_dynamic("ma_u16", ma_u16)?;
    exec.insert_dynamic("mb_u16", mb_u16)?;
    exec.insert_dynamic("a_u32", a_u32)?;
    exec.insert_dynamic("b_u32", b_u32)?;
    exec.insert_dynamic("ma_u32", ma_u32)?;
    exec.insert_dynamic("mb_u32", mb_u32)?;
    exec.insert_dynamic("a_u64", a_u64)?;
    exec.insert_dynamic("b_u64", b_u64)?;
    exec.insert_dynamic("ma_u64", ma_u64)?;
    exec.insert_dynamic("mb_u64", mb_u64)?;
    if !is_vulkan {
        exec.insert_dynamic("a_f16", a_f16)?;
        exec.insert_dynamic("b_f16", b_f16)?;
        exec.insert_dynamic("ma_f16", ma_f16)?;
        exec.insert_dynamic("mb_f16", mb_f16)?;
    }
    exec.insert_dynamic("a_f32", a_f32)?;
    exec.insert_dynamic("b_f32", b_f32)?;
    exec.insert_dynamic("ma_f32", ma_f32)?;
    exec.insert_dynamic("mb_f32", mb_f32)?;
    if !is_vulkan {
        exec.insert_dynamic("a_f64", a_f64)?;
        exec.insert_dynamic("b_f64", b_f64)?;
        exec.insert_dynamic("ma_f64", ma_f64)?;
        exec.insert_dynamic("mb_f64", mb_f64)?;
    }
    exec.insert_dynamic("a_bool", a_bool)?;
    exec.insert_dynamic("b_bool", b_bool)?;
    exec.insert_dynamic("ma_bool", ma_bool)?;
    exec.insert_dynamic("mb_bool", mb_bool)?;
    if !is_vulkan {
        exec.insert_dynamic("a_bitset", a_bitset)?;
        exec.insert_dynamic("b_bitset", b_bitset)?;
        exec.insert_dynamic("ma_bitset", ma_bitset)?;
        exec.insert_dynamic("mb_bitset", mb_bitset)?;
    }

    exec.step()?;

    print_group::<i8>(&mut exec, "i8", n, false, true, true)?;
    print_group::<i16>(&mut exec, "i16", n, false, true, true)?;
    print_group::<i32>(&mut exec, "i32", n, false, true, true)?;
    print_group::<i64>(&mut exec, "i64", n, false, true, true)?;
    print_group::<u8>(&mut exec, "u8", n, false, false, false)?;
    print_group::<u16>(&mut exec, "u16", n, false, false, false)?;
    print_group::<u32>(&mut exec, "u32", n, false, false, false)?;
    print_group::<u64>(&mut exec, "u64", n, false, false, false)?;
    if !is_vulkan {
        print_group::<F16>(&mut exec, "f16", n, true, true, true)?;
    }
    print_group::<f32>(&mut exec, "f32", n, true, true, true)?;
    if !is_vulkan {
        print_group::<f64>(&mut exec, "f64", n, true, true, true)?;
    }
    print_group::<bool>(&mut exec, "bool", n, false, false, false)?;
    if !is_vulkan {
        print_group::<Bitset>(&mut exec, "bitset", n, false, false, false)?;
    }

    println!("ops_matrix completed");

    Ok(())
}
