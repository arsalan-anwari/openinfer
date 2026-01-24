use anyhow::Result;
use openinfer::{
    graph, Bitset, Device, Executor, F16, ModelLoader, Simulator, Tensor, TensorElement,
    TensorOptions, BF16, F8E5M2, I1, I2, I4, U1, U2, U4,
};
use openinfer::{format_truncated, FormatValue};
use std::collections::HashMap;
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

fn pack_signed(bits: u8, values: &[i8]) -> Vec<u8> {
    let per = 8 / bits;
    let mask = (1u16 << bits) as u8 - 1;
    let storage_len = (values.len() + per as usize - 1) / per as usize;
    let mut out = vec![0u8; storage_len];
    for (idx, value) in values.iter().enumerate() {
        let raw = (*value as i16 as u8) & mask;
        let byte_idx = idx / per as usize;
        let shift = (idx % per as usize) as u8 * bits;
        out[byte_idx] = (out[byte_idx] & !(mask << shift)) | ((raw & mask) << shift);
    }
    out
}

fn pack_unsigned(bits: u8, values: &[u8]) -> Vec<u8> {
    let per = 8 / bits;
    let mask = (1u16 << bits) as u8 - 1;
    let storage_len = (values.len() + per as usize - 1) / per as usize;
    let mut out = vec![0u8; storage_len];
    for (idx, value) in values.iter().enumerate() {
        let raw = *value & mask;
        let byte_idx = idx / per as usize;
        let shift = (idx % per as usize) as u8 * bits;
        out[byte_idx] = (out[byte_idx] & !(mask << shift)) | ((raw & mask) << shift);
    }
    out
}

fn tensor_packed_i4(values: &[i8], shape: Vec<usize>) -> Result<Tensor<I4>> {
    let packed = pack_signed(4, values);
    let data = packed.into_iter().map(|bits| I4 { bits }).collect();
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn tensor_packed_i2(values: &[i8], shape: Vec<usize>) -> Result<Tensor<I2>> {
    let packed = pack_signed(2, values);
    let data = packed.into_iter().map(|bits| I2 { bits }).collect();
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn tensor_packed_i1(values: &[i8], shape: Vec<usize>) -> Result<Tensor<I1>> {
    let packed = pack_signed(1, values);
    let data = packed.into_iter().map(|bits| I1 { bits }).collect();
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn tensor_packed_u4(values: &[u8], shape: Vec<usize>) -> Result<Tensor<U4>> {
    let packed = pack_unsigned(4, values);
    let data = packed.into_iter().map(|bits| U4 { bits }).collect();
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn tensor_packed_u2(values: &[u8], shape: Vec<usize>) -> Result<Tensor<U2>> {
    let packed = pack_unsigned(2, values);
    let data = packed.into_iter().map(|bits| U2 { bits }).collect();
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn tensor_packed_u1(values: &[u8], shape: Vec<usize>) -> Result<Tensor<U1>> {
    let packed = pack_unsigned(1, values);
    let data = packed.into_iter().map(|bits| U1 { bits }).collect();
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn insert<T: TensorElement>(exec: &mut Executor, name: &str, tensor: Tensor<T>) -> Result<()> {
    exec.insert_dynamic(name, <T as TensorElement>::into_value(tensor))
}

fn format_tensor<T: TensorElement + FormatValue>(exec: &mut Executor, name: &str) -> Result<String> {
    let tensor: Tensor<T> = exec.fetch(name)?;
    let limit = 10.min(tensor.data.len());
    Ok(format!("{}", format_truncated(&tensor.data[..limit])))
}

trait ToF64 {
    fn to_f64(self) -> f64;
}

impl ToF64 for F16 {
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
}

impl ToF64 for BF16 {
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
}

impl ToF64 for F8E5M2 {
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
}

impl ToF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

#[derive(Clone, Copy)]
struct FloatTol {
    abs: f64,
    rel: f64,
}

impl FloatTol {
    fn f16() -> Self {
        Self { abs: 0.6, rel: 0.08 }
    }

    fn bf16() -> Self {
        Self { abs: 0.1, rel: 0.02 }
    }

    fn f8() -> Self {
        Self { abs: 0.6, rel: 0.25 }
    }

    fn f32() -> Self {
        Self { abs: 1e-4, rel: 1e-4 }
    }

    fn f64() -> Self {
        Self { abs: 1e-8, rel: 1e-8 }
    }
}

fn within_tol(a: f64, b: f64, tol: FloatTol) -> bool {
    let diff = (a - b).abs();
    if diff <= tol.abs {
        return true;
    }
    let scale = a.abs().max(b.abs());
    diff <= tol.rel * scale
}

fn float_status(current: &[f64], reference: &[f64], tol: FloatTol) -> &'static str {
    if current.len() != reference.len() {
        return "❌";
    }
    let mut exact = true;
    for (a, b) in current.iter().zip(reference.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        if a == b {
            continue;
        }
        exact = false;
        if !within_tol(*a, *b, tol) {
            return "❌";
        }
    }
    if exact { "✅" } else { "⚠️" }
}

fn collect_tensor_ref<T: TensorElement + FormatValue>(
    refs: &mut HashMap<String, String>,
    exec: &mut Executor,
    name: &str,
) -> Result<()> {
    let formatted = format_tensor::<T>(exec, name)?;
    refs.insert(name.to_string(), formatted);
    Ok(())
}

fn collect_tensor_ref_float<T: TensorElement + FormatValue + Copy + ToF64>(
    refs: &mut HashMap<String, String>,
    float_refs: &mut HashMap<String, Vec<f64>>,
    exec: &mut Executor,
    name: &str,
) -> Result<()> {
    collect_tensor_ref::<T>(refs, exec, name)?;
    let tensor: Tensor<T> = exec.fetch(name)?;
    let values = tensor.data.iter().map(|v| v.to_f64()).collect();
    float_refs.insert(name.to_string(), values);
    Ok(())
}

fn validate_tensor<T: TensorElement + FormatValue>(
    refs: &HashMap<String, String>,
    exec: &mut Executor,
    name: &str,
) -> Result<()> {
    let formatted = format_tensor::<T>(exec, name)?;
    let ref_val = refs.get(name).cloned().unwrap_or_else(|| "<missing>".to_string());
    let status = if formatted == ref_val { "✅" } else { "❌" };
    log::info!("[{}] {} = {} -- ref = {}", status, name, formatted, ref_val);
    Ok(())
}

fn validate_tensor_float<T: TensorElement + FormatValue + Copy + ToF64>(
    refs: &HashMap<String, String>,
    float_refs: &HashMap<String, Vec<f64>>,
    exec: &mut Executor,
    name: &str,
    tol: FloatTol,
) -> Result<()> {
    let formatted = format_tensor::<T>(exec, name)?;
    let ref_val = refs.get(name).cloned().unwrap_or_else(|| "<missing>".to_string());
    let status = match float_refs.get(name) {
        Some(reference) => {
            let tensor: Tensor<T> = exec.fetch(name)?;
            let current: Vec<f64> = tensor.data.iter().map(|v| v.to_f64()).collect();
            float_status(&current, reference, tol)
        }
        None => "❌",
    };
    log::info!("[{}] {} = {} -- ref = {}", status, name, formatted, ref_val);
    Ok(())
}

fn collect_named<T: TensorElement + FormatValue>(
    refs: &mut HashMap<String, String>,
    exec: &mut Executor,
    names: &[&str],
) -> Result<()> {
    for name in names {
        collect_tensor_ref::<T>(refs, exec, name)?;
    }
    Ok(())
}

fn collect_named_float<T: TensorElement + FormatValue + Copy + ToF64>(
    refs: &mut HashMap<String, String>,
    float_refs: &mut HashMap<String, Vec<f64>>,
    exec: &mut Executor,
    names: &[&str],
) -> Result<()> {
    for name in names {
        collect_tensor_ref_float::<T>(refs, float_refs, exec, name)?;
    }
    Ok(())
}

fn validate_named<T: TensorElement + FormatValue>(
    refs: &HashMap<String, String>,
    exec: &mut Executor,
    names: &[&str],
) -> Result<()> {
    for name in names {
        validate_tensor::<T>(refs, exec, name)?;
    }
    Ok(())
}

fn validate_named_float<T: TensorElement + FormatValue + Copy + ToF64>(
    refs: &HashMap<String, String>,
    float_refs: &HashMap<String, Vec<f64>>,
    exec: &mut Executor,
    names: &[&str],
    tol: FloatTol,
) -> Result<()> {
    for name in names {
        validate_tensor_float::<T>(refs, float_refs, exec, name, tol)?;
    }
    Ok(())
}

const I8_OUTPUTS: &[&str] = &[
    "add_i8",
    "add_b_i8",
    "mul_i8",
    "mul_b_i8",
    "in_add_i8",
    "in_add_b_i8",
    "in_mul_i8",
    "in_mul_b_i8",
    "mm_i8",
    "mm_b_i8",
    "in_mm_i8",
    "in_mm_b_i8",
    "add_acc_i4",
    "add_acc_b_i4",
    "add_acc_i2",
    "add_acc_b_i2",
    "add_acc_i1",
    "add_acc_b_i1",
    "mul_acc_i4",
    "mul_acc_b_i4",
    "mul_acc_i2",
    "mul_acc_b_i2",
    "mul_acc_i1",
    "mul_acc_b_i1",
];

const I16_OUTPUTS: &[&str] = &[
    "add_i16",
    "add_b_i16",
    "mul_i16",
    "mul_b_i16",
    "in_add_i16",
    "in_add_b_i16",
    "in_mul_i16",
    "in_mul_b_i16",
    "mm_i16",
    "mm_b_i16",
    "in_mm_i16",
    "in_mm_b_i16",
    "add_acc_i8",
    "add_acc_b_i8",
    "mul_acc_i8",
    "mul_acc_b_i8",
    "mm_acc_i8",
    "mm_acc_b_i8",
];

const I32_OUTPUTS: &[&str] = &[
    "add_i32",
    "add_b_i32",
    "mul_i32",
    "mul_b_i32",
    "in_add_i32",
    "in_add_b_i32",
    "in_mul_i32",
    "in_mul_b_i32",
    "mm_i32",
    "mm_b_i32",
    "in_mm_i32",
    "in_mm_b_i32",
    "add_acc_i16",
    "add_acc_b_i16",
    "mul_acc_i16",
    "mul_acc_b_i16",
    "mm_acc_i16",
    "mm_acc_b_i16",
];

const I64_OUTPUTS: &[&str] = &[
    "add_i64",
    "add_b_i64",
    "mul_i64",
    "mul_b_i64",
    "in_add_i64",
    "in_add_b_i64",
    "in_mul_i64",
    "in_mul_b_i64",
    "mm_i64",
    "mm_b_i64",
    "in_mm_i64",
    "in_mm_b_i64",
    "add_acc_i32",
    "add_acc_b_i32",
    "mul_acc_i32",
    "mul_acc_b_i32",
    "mm_acc_i32",
    "mm_acc_b_i32",
];

const U8_OUTPUTS: &[&str] = &[
    "add_u8",
    "add_b_u8",
    "mul_u8",
    "mul_b_u8",
    "in_add_u8",
    "in_add_b_u8",
    "in_mul_u8",
    "in_mul_b_u8",
    "mm_u8",
    "mm_b_u8",
    "in_mm_u8",
    "in_mm_b_u8",
    "add_acc_u4",
    "add_acc_b_u4",
    "add_acc_u2",
    "add_acc_b_u2",
    "add_acc_u1",
    "add_acc_b_u1",
    "mul_acc_u4",
    "mul_acc_b_u4",
    "mul_acc_u2",
    "mul_acc_b_u2",
    "mul_acc_u1",
    "mul_acc_b_u1",
];

const U16_OUTPUTS: &[&str] = &[
    "add_u16",
    "add_b_u16",
    "mul_u16",
    "mul_b_u16",
    "in_add_u16",
    "in_add_b_u16",
    "in_mul_u16",
    "in_mul_b_u16",
    "mm_u16",
    "mm_b_u16",
    "in_mm_u16",
    "in_mm_b_u16",
    "add_acc_u8",
    "add_acc_b_u8",
    "mul_acc_u8",
    "mul_acc_b_u8",
    "mm_acc_u8",
    "mm_acc_b_u8",
];

const U32_OUTPUTS: &[&str] = &[
    "add_u32",
    "add_b_u32",
    "mul_u32",
    "mul_b_u32",
    "in_add_u32",
    "in_add_b_u32",
    "in_mul_u32",
    "in_mul_b_u32",
    "mm_u32",
    "mm_b_u32",
    "in_mm_u32",
    "in_mm_b_u32",
    "add_acc_u16",
    "add_acc_b_u16",
    "mul_acc_u16",
    "mul_acc_b_u16",
    "mm_acc_u16",
    "mm_acc_b_u16",
];

const U64_OUTPUTS: &[&str] = &[
    "add_u64",
    "add_b_u64",
    "mul_u64",
    "mul_b_u64",
    "in_add_u64",
    "in_add_b_u64",
    "in_mul_u64",
    "in_mul_b_u64",
    "mm_u64",
    "mm_b_u64",
    "in_mm_u64",
    "in_mm_b_u64",
    "add_acc_u32",
    "add_acc_b_u32",
    "mul_acc_u32",
    "mul_acc_b_u32",
    "mm_acc_u32",
    "mm_acc_b_u32",
];

const BOOL_OUTPUTS: &[&str] = &[
    "add_bool",
    "add_b_bool",
    "mul_bool",
    "mul_b_bool",
    "in_add_bool",
    "in_add_b_bool",
    "in_mul_bool",
    "in_mul_b_bool",
    "mm_bool",
    "mm_b_bool",
    "in_mm_bool",
    "in_mm_b_bool",
];

const BITSET_OUTPUTS: &[&str] = &[
    "add_bitset",
    "add_b_bitset",
    "mul_bitset",
    "mul_b_bitset",
    "in_add_bitset",
    "in_add_b_bitset",
    "in_mul_bitset",
    "in_mul_b_bitset",
    "mm_bitset",
    "mm_b_bitset",
    "in_mm_bitset",
    "in_mm_b_bitset",
];

const I4_OUTPUTS: &[&str] = &[
    "add_i4",
    "add_b_i4",
    "mul_i4",
    "mul_b_i4",
    "in_add_i4",
    "in_add_b_i4",
    "in_mul_i4",
    "in_mul_b_i4",
];

const I2_OUTPUTS: &[&str] = &[
    "add_i2",
    "add_b_i2",
    "mul_i2",
    "mul_b_i2",
    "in_add_i2",
    "in_add_b_i2",
    "in_mul_i2",
    "in_mul_b_i2",
];

const I1_OUTPUTS: &[&str] = &[
    "add_i1",
    "add_b_i1",
    "mul_i1",
    "mul_b_i1",
    "in_add_i1",
    "in_add_b_i1",
    "in_mul_i1",
    "in_mul_b_i1",
];

const U4_OUTPUTS: &[&str] = &[
    "add_u4",
    "add_b_u4",
    "mul_u4",
    "mul_b_u4",
    "in_add_u4",
    "in_add_b_u4",
    "in_mul_u4",
    "in_mul_b_u4",
];

const U2_OUTPUTS: &[&str] = &[
    "add_u2",
    "add_b_u2",
    "mul_u2",
    "mul_b_u2",
    "in_add_u2",
    "in_add_b_u2",
    "in_mul_u2",
    "in_mul_b_u2",
];

const U1_OUTPUTS: &[&str] = &[
    "add_u1",
    "add_b_u1",
    "mul_u1",
    "mul_b_u1",
    "in_add_u1",
    "in_add_b_u1",
    "in_mul_u1",
    "in_mul_b_u1",
];

const F16_OUTPUTS: &[&str] = &[
    "add_f16",
    "add_b_f16",
    "mul_f16",
    "mul_b_f16",
    "in_add_f16",
    "in_add_b_f16",
    "in_mul_f16",
    "in_mul_b_f16",
    "mm_f16",
    "mm_b_f16",
    "in_mm_f16",
    "in_mm_b_f16",
];

const BF16_OUTPUTS: &[&str] = &[
    "add_bf16",
    "add_b_bf16",
    "mul_bf16",
    "mul_b_bf16",
    "in_add_bf16",
    "in_add_b_bf16",
    "in_mul_bf16",
    "in_mul_b_bf16",
];

const F8_OUTPUTS: &[&str] = &[
    "add_f8",
    "add_b_f8",
    "mul_f8",
    "mul_b_f8",
    "in_add_f8",
    "in_add_b_f8",
    "in_mul_f8",
    "in_mul_b_f8",
];

const F32_OUTPUTS: &[&str] = &[
    "add_f32",
    "add_b_f32",
    "mul_f32",
    "mul_b_f32",
    "in_add_f32",
    "in_add_b_f32",
    "in_mul_f32",
    "in_mul_b_f32",
    "mm_f32",
    "mm_b_f32",
    "in_mm_f32",
    "in_mm_b_f32",
];

const F64_OUTPUTS: &[&str] = &[
    "add_f64",
    "add_b_f64",
    "mul_f64",
    "mul_b_f64",
    "in_add_f64",
    "in_add_b_f64",
    "in_mul_f64",
    "in_mul_b_f64",
    "mm_f64",
    "mm_b_f64",
    "in_mm_f64",
    "in_mm_b_f64",
];

fn populate_exec(exec: &mut Executor, v: usize, s: usize, m: usize, k: usize, n: usize, b: usize) -> Result<()> {
    let i8_vals: Vec<i8> = (0..v).map(|i| i as i8 - 4).collect();
    let i8_vals_b: Vec<i8> = (0..v).map(|i| i as i8 - 1).collect();
    let i8_scalar = vec![i8_vals_b[0]];
    let u8_vals: Vec<u8> = (0..v).map(|i| i as u8).collect();
    let u8_vals_b: Vec<u8> = (0..v).map(|i| (i as u8).wrapping_add(1)).collect();
    let u8_scalar = vec![u8_vals_b[0]];

    insert(exec, "a_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;
    insert(exec, "b_i8", tensor_with_shape(i8_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_i8_b", tensor_with_shape(i8_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;

    insert(exec, "a_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "b_i16", tensor_with_shape(i8_vals_b.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "b_i16_b", tensor_with_shape(i8_scalar.iter().map(|v| *v as i16).collect(), vec![s])?)?;
    insert(exec, "in_add_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "in_add_b_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "in_mul_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "in_mul_b_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;

    insert(exec, "a_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "b_i32", tensor_with_shape(i8_vals_b.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "b_i32_b", tensor_with_shape(i8_scalar.iter().map(|v| *v as i32).collect(), vec![s])?)?;
    insert(exec, "in_add_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "in_add_b_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "in_mul_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "in_mul_b_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;

    insert(exec, "a_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "b_i64", tensor_with_shape(i8_vals_b.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "b_i64_b", tensor_with_shape(i8_scalar.iter().map(|v| *v as i64).collect(), vec![s])?)?;
    insert(exec, "in_add_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "in_add_b_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "in_mul_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "in_mul_b_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;

    insert(exec, "a_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;
    insert(exec, "b_u8", tensor_with_shape(u8_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_u8_b", tensor_with_shape(u8_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;

    insert(exec, "a_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "b_u16", tensor_with_shape(u8_vals_b.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "b_u16_b", tensor_with_shape(u8_scalar.iter().map(|v| *v as u16).collect(), vec![s])?)?;
    insert(exec, "in_add_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "in_add_b_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "in_mul_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "in_mul_b_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;

    insert(exec, "a_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "b_u32", tensor_with_shape(u8_vals_b.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "b_u32_b", tensor_with_shape(u8_scalar.iter().map(|v| *v as u32).collect(), vec![s])?)?;
    insert(exec, "in_add_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "in_add_b_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "in_mul_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "in_mul_b_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;

    insert(exec, "a_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "b_u64", tensor_with_shape(u8_vals_b.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "b_u64_b", tensor_with_shape(u8_scalar.iter().map(|v| *v as u64).collect(), vec![s])?)?;
    insert(exec, "in_add_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "in_add_b_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "in_mul_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "in_mul_b_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;

    let f16_vals: Vec<F16> = (0..v).map(|i| F16::from_f32(i as f32 * 0.5 - 2.0)).collect();
    let f16_vals_b: Vec<F16> = (0..v).map(|i| F16::from_f32(i as f32 * 0.5 + 0.5)).collect();
    let f16_scalar = vec![f16_vals_b[0]];
    let bf16_vals: Vec<BF16> = (0..v).map(|i| BF16::from_f32(i as f32 * 0.5 - 2.0)).collect();
    let bf16_vals_b: Vec<BF16> = (0..v).map(|i| BF16::from_f32(i as f32 * 0.5 + 0.5)).collect();
    let bf16_scalar = vec![bf16_vals_b[0]];
    let f8_vals: Vec<F8E5M2> = (0..v).map(|i| F8E5M2::from_f32(i as f32 * 0.5 - 2.0)).collect();
    let f8_vals_b: Vec<F8E5M2> = (0..v).map(|i| F8E5M2::from_f32(i as f32 * 0.5 + 0.5)).collect();
    let f8_scalar = vec![f8_vals_b[0]];
    let f32_vals: Vec<f32> = (0..v).map(|i| i as f32 * 0.5 - 2.0).collect();
    let f32_vals_b: Vec<f32> = (0..v).map(|i| i as f32 * 0.5 + 0.5).collect();
    let f32_scalar = vec![f32_vals_b[0]];
    let f64_vals: Vec<f64> = (0..v).map(|i| i as f64 * 0.5 - 2.0).collect();
    let f64_vals_b: Vec<f64> = (0..v).map(|i| i as f64 * 0.5 + 0.5).collect();
    let f64_scalar = vec![f64_vals_b[0]];

    insert(exec, "a_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;
    insert(exec, "b_f16", tensor_with_shape(f16_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_f16_b", tensor_with_shape(f16_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;

    insert(exec, "a_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;
    insert(exec, "b_bf16", tensor_with_shape(bf16_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_bf16_b", tensor_with_shape(bf16_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;

    insert(exec, "a_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;
    insert(exec, "b_f8", tensor_with_shape(f8_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_f8_b", tensor_with_shape(f8_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;

    insert(exec, "a_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;
    insert(exec, "b_f32", tensor_with_shape(f32_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_f32_b", tensor_with_shape(f32_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;

    insert(exec, "a_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;
    insert(exec, "b_f64", tensor_with_shape(f64_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_f64_b", tensor_with_shape(f64_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;

    let bool_vals: Vec<bool> = (0..v).map(|i| i % 2 == 0).collect();
    let bool_vals_b: Vec<bool> = (0..v).map(|i| i % 3 == 0).collect();
    let bool_scalar = vec![bool_vals_b[0]];
    insert(exec, "a_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;
    insert(exec, "b_bool", tensor_with_shape(bool_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_bool_b", tensor_with_shape(bool_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;

    let bitset_vals: Vec<Bitset> = (0..v)
        .map(|i| Bitset {
            bits: (i as u8).wrapping_mul(3),
        })
        .collect();
    let bitset_vals_b: Vec<Bitset> = (0..v)
        .map(|i| Bitset {
            bits: (i as u8).wrapping_mul(5),
        })
        .collect();
    let bitset_scalar = vec![bitset_vals_b[0]];
    insert(exec, "a_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;
    insert(exec, "b_bitset", tensor_with_shape(bitset_vals_b.clone(), vec![v])?)?;
    insert(exec, "b_bitset_b", tensor_with_shape(bitset_scalar.clone(), vec![s])?)?;
    insert(exec, "in_add_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;
    insert(exec, "in_add_b_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;
    insert(exec, "in_mul_b_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;

    let i4_vals = vec![-8, -4, -1, 0, 1, 3, 6, 7];
    let i4_vals_b = vec![1, -1, 2, -2, 3, -3, 4, -4];
    insert(exec, "a_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;
    insert(exec, "b_i4", tensor_packed_i4(&i4_vals_b, vec![v])?)?;
    insert(exec, "b_i4_b", tensor_packed_i4(&[i4_vals_b[0]], vec![s])?)?;
    insert(exec, "in_add_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;
    insert(exec, "in_add_b_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;
    insert(exec, "in_mul_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;
    insert(exec, "in_mul_b_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;

    let i2_vals = vec![-2, -1, 0, 1, -2, -1, 0, 1];
    let i2_vals_b = vec![1, 0, -1, -2, 1, 0, -1, -2];
    insert(exec, "a_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;
    insert(exec, "b_i2", tensor_packed_i2(&i2_vals_b, vec![v])?)?;
    insert(exec, "b_i2_b", tensor_packed_i2(&[i2_vals_b[0]], vec![s])?)?;
    insert(exec, "in_add_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;
    insert(exec, "in_add_b_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;
    insert(exec, "in_mul_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;
    insert(exec, "in_mul_b_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;

    let i1_vals = vec![-1, 0, -1, 0, -1, 0, -1, 0];
    let i1_vals_b = vec![0, -1, 0, -1, 0, -1, 0, -1];
    insert(exec, "a_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;
    insert(exec, "b_i1", tensor_packed_i1(&i1_vals_b, vec![v])?)?;
    insert(exec, "b_i1_b", tensor_packed_i1(&[i1_vals_b[0]], vec![s])?)?;
    insert(exec, "in_add_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;
    insert(exec, "in_add_b_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;
    insert(exec, "in_mul_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;
    insert(exec, "in_mul_b_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;

    let u4_vals = vec![0, 1, 2, 3, 4, 5, 14, 15];
    let u4_vals_b = vec![1, 2, 3, 4, 0, 1, 2, 3];
    insert(exec, "a_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;
    insert(exec, "b_u4", tensor_packed_u4(&u4_vals_b, vec![v])?)?;
    insert(exec, "b_u4_b", tensor_packed_u4(&[u4_vals_b[0]], vec![s])?)?;
    insert(exec, "in_add_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;
    insert(exec, "in_add_b_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;
    insert(exec, "in_mul_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;
    insert(exec, "in_mul_b_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;

    let u2_vals = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let u2_vals_b = vec![3, 2, 1, 0, 3, 2, 1, 0];
    insert(exec, "a_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;
    insert(exec, "b_u2", tensor_packed_u2(&u2_vals_b, vec![v])?)?;
    insert(exec, "b_u2_b", tensor_packed_u2(&[u2_vals_b[0]], vec![s])?)?;
    insert(exec, "in_add_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;
    insert(exec, "in_add_b_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;
    insert(exec, "in_mul_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;
    insert(exec, "in_mul_b_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;

    let u1_vals = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let u1_vals_b = vec![1, 0, 1, 0, 0, 1, 0, 1];
    insert(exec, "a_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;
    insert(exec, "b_u1", tensor_packed_u1(&u1_vals_b, vec![v])?)?;
    insert(exec, "b_u1_b", tensor_packed_u1(&[u1_vals_b[0]], vec![s])?)?;
    insert(exec, "in_add_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;
    insert(exec, "in_add_b_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;
    insert(exec, "in_mul_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;
    insert(exec, "in_mul_b_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;

    let mk = m * k;
    let kn = k * n;
    let batch_mk = b * mk;
    let batch_kn = b * kn;

    let ma_i8 = tensor_with_shape((0..batch_mk).map(|i| i as i8 - 2).collect(), vec![b, m, k])?;
    let mb_i8 = tensor_with_shape((0..batch_kn).map(|i| i as i8 - 1).collect(), vec![b, k, n])?;
    let mb_i8_b = tensor_with_shape((0..kn).map(|i| i as i8 - 1).collect(), vec![s, k, n])?;
    insert(exec, "ma_i8", ma_i8.clone())?;
    insert(exec, "mb_i8", mb_i8.clone())?;
    insert(exec, "mb_i8_b", mb_i8_b)?;
    insert(exec, "in_mm_i8", ma_i8.clone())?;
    insert(exec, "in_mm_b_i8", ma_i8)?;

    let ma_i16 = tensor_with_shape((0..batch_mk).map(|i| i as i16 - 2).collect(), vec![b, m, k])?;
    let mb_i16 = tensor_with_shape((0..batch_kn).map(|i| i as i16 - 1).collect(), vec![b, k, n])?;
    let mb_i16_b = tensor_with_shape((0..kn).map(|i| i as i16 - 1).collect(), vec![s, k, n])?;
    insert(exec, "ma_i16", ma_i16.clone())?;
    insert(exec, "mb_i16", mb_i16.clone())?;
    insert(exec, "mb_i16_b", mb_i16_b)?;
    insert(exec, "in_mm_i16", ma_i16.clone())?;
    insert(exec, "in_mm_b_i16", ma_i16)?;

    let ma_i32 = tensor_with_shape((0..batch_mk).map(|i| i as i32 - 2).collect(), vec![b, m, k])?;
    let mb_i32 = tensor_with_shape((0..batch_kn).map(|i| i as i32 - 1).collect(), vec![b, k, n])?;
    let mb_i32_b = tensor_with_shape((0..kn).map(|i| i as i32 - 1).collect(), vec![s, k, n])?;
    insert(exec, "ma_i32", ma_i32.clone())?;
    insert(exec, "mb_i32", mb_i32.clone())?;
    insert(exec, "mb_i32_b", mb_i32_b)?;
    insert(exec, "in_mm_i32", ma_i32.clone())?;
    insert(exec, "in_mm_b_i32", ma_i32)?;

    let ma_i64 = tensor_with_shape((0..batch_mk).map(|i| i as i64 - 2).collect(), vec![b, m, k])?;
    let mb_i64 = tensor_with_shape((0..batch_kn).map(|i| i as i64 - 1).collect(), vec![b, k, n])?;
    let mb_i64_b = tensor_with_shape((0..kn).map(|i| i as i64 - 1).collect(), vec![s, k, n])?;
    insert(exec, "ma_i64", ma_i64.clone())?;
    insert(exec, "mb_i64", mb_i64.clone())?;
    insert(exec, "mb_i64_b", mb_i64_b)?;
    insert(exec, "in_mm_i64", ma_i64.clone())?;
    insert(exec, "in_mm_b_i64", ma_i64)?;

    let ma_u8 = tensor_with_shape((0..batch_mk).map(|i| i as u8).collect(), vec![b, m, k])?;
    let mb_u8 = tensor_with_shape((0..batch_kn).map(|i| (i as u8).wrapping_add(1)).collect(), vec![b, k, n])?;
    let mb_u8_b = tensor_with_shape((0..kn).map(|i| (i as u8).wrapping_add(1)).collect(), vec![s, k, n])?;
    insert(exec, "ma_u8", ma_u8.clone())?;
    insert(exec, "mb_u8", mb_u8.clone())?;
    insert(exec, "mb_u8_b", mb_u8_b)?;
    insert(exec, "in_mm_u8", ma_u8.clone())?;
    insert(exec, "in_mm_b_u8", ma_u8)?;

    let ma_u16 = tensor_with_shape((0..batch_mk).map(|i| i as u16).collect(), vec![b, m, k])?;
    let mb_u16 = tensor_with_shape((0..batch_kn).map(|i| (i as u16).wrapping_add(1)).collect(), vec![b, k, n])?;
    let mb_u16_b = tensor_with_shape((0..kn).map(|i| (i as u16).wrapping_add(1)).collect(), vec![s, k, n])?;
    insert(exec, "ma_u16", ma_u16.clone())?;
    insert(exec, "mb_u16", mb_u16.clone())?;
    insert(exec, "mb_u16_b", mb_u16_b)?;
    insert(exec, "in_mm_u16", ma_u16.clone())?;
    insert(exec, "in_mm_b_u16", ma_u16)?;

    let ma_u32 = tensor_with_shape((0..batch_mk).map(|i| i as u32).collect(), vec![b, m, k])?;
    let mb_u32 = tensor_with_shape((0..batch_kn).map(|i| (i as u32).wrapping_add(1)).collect(), vec![b, k, n])?;
    let mb_u32_b = tensor_with_shape((0..kn).map(|i| (i as u32).wrapping_add(1)).collect(), vec![s, k, n])?;
    insert(exec, "ma_u32", ma_u32.clone())?;
    insert(exec, "mb_u32", mb_u32.clone())?;
    insert(exec, "mb_u32_b", mb_u32_b)?;
    insert(exec, "in_mm_u32", ma_u32.clone())?;
    insert(exec, "in_mm_b_u32", ma_u32)?;

    let ma_u64 = tensor_with_shape((0..batch_mk).map(|i| i as u64).collect(), vec![b, m, k])?;
    let mb_u64 = tensor_with_shape((0..batch_kn).map(|i| (i as u64).wrapping_add(1)).collect(), vec![b, k, n])?;
    let mb_u64_b = tensor_with_shape((0..kn).map(|i| (i as u64).wrapping_add(1)).collect(), vec![s, k, n])?;
    insert(exec, "ma_u64", ma_u64.clone())?;
    insert(exec, "mb_u64", mb_u64.clone())?;
    insert(exec, "mb_u64_b", mb_u64_b)?;
    insert(exec, "in_mm_u64", ma_u64.clone())?;
    insert(exec, "in_mm_b_u64", ma_u64)?;

    let ma_f16 = tensor_with_shape(
        (0..batch_mk).map(|i| F16::from_f32(i as f32 * 0.25 - 1.0)).collect(),
        vec![b, m, k],
    )?;
    let mb_f16 = tensor_with_shape(
        (0..batch_kn).map(|i| F16::from_f32(i as f32 * 0.25 + 0.5)).collect(),
        vec![b, k, n],
    )?;
    let mb_f16_b = tensor_with_shape(
        (0..kn).map(|i| F16::from_f32(i as f32 * 0.25 + 0.5)).collect(),
        vec![s, k, n],
    )?;
    insert(exec, "ma_f16", ma_f16.clone())?;
    insert(exec, "mb_f16", mb_f16.clone())?;
    insert(exec, "mb_f16_b", mb_f16_b)?;
    insert(exec, "in_mm_f16", ma_f16.clone())?;
    insert(exec, "in_mm_b_f16", ma_f16)?;

    let ma_f32 = tensor_with_shape(
        (0..batch_mk).map(|i| i as f32 * 0.25 - 1.0).collect(),
        vec![b, m, k],
    )?;
    let mb_f32 = tensor_with_shape(
        (0..batch_kn).map(|i| i as f32 * 0.25 + 0.5).collect(),
        vec![b, k, n],
    )?;
    let mb_f32_b = tensor_with_shape(
        (0..kn).map(|i| i as f32 * 0.25 + 0.5).collect(),
        vec![s, k, n],
    )?;
    insert(exec, "ma_f32", ma_f32.clone())?;
    insert(exec, "mb_f32", mb_f32.clone())?;
    insert(exec, "mb_f32_b", mb_f32_b)?;
    insert(exec, "in_mm_f32", ma_f32.clone())?;
    insert(exec, "in_mm_b_f32", ma_f32)?;

    let ma_f64 = tensor_with_shape(
        (0..batch_mk).map(|i| i as f64 * 0.25 - 1.0).collect(),
        vec![b, m, k],
    )?;
    let mb_f64 = tensor_with_shape(
        (0..batch_kn).map(|i| i as f64 * 0.25 + 0.5).collect(),
        vec![b, k, n],
    )?;
    let mb_f64_b = tensor_with_shape(
        (0..kn).map(|i| i as f64 * 0.25 + 0.5).collect(),
        vec![s, k, n],
    )?;
    insert(exec, "ma_f64", ma_f64.clone())?;
    insert(exec, "mb_f64", mb_f64.clone())?;
    insert(exec, "mb_f64_b", mb_f64_b)?;
    insert(exec, "in_mm_f64", ma_f64.clone())?;
    insert(exec, "in_mm_b_f64", ma_f64)?;

    let ma_bool = tensor_with_shape((0..batch_mk).map(|i| i % 2 == 0).collect(), vec![b, m, k])?;
    let mb_bool = tensor_with_shape((0..batch_kn).map(|i| i % 3 == 0).collect(), vec![b, k, n])?;
    let mb_bool_b = tensor_with_shape((0..kn).map(|i| i % 3 == 0).collect(), vec![s, k, n])?;
    insert(exec, "ma_bool", ma_bool.clone())?;
    insert(exec, "mb_bool", mb_bool.clone())?;
    insert(exec, "mb_bool_b", mb_bool_b)?;
    insert(exec, "in_mm_bool", ma_bool.clone())?;
    insert(exec, "in_mm_b_bool", ma_bool)?;

    let ma_bitset = tensor_with_shape(
        (0..batch_mk).map(|i| Bitset { bits: (i as u8).wrapping_mul(2) }).collect(),
        vec![b, m, k],
    )?;
    let mb_bitset = tensor_with_shape(
        (0..batch_kn).map(|i| Bitset { bits: (i as u8).wrapping_mul(7) }).collect(),
        vec![b, k, n],
    )?;
    let mb_bitset_b = tensor_with_shape(
        (0..kn).map(|i| Bitset { bits: (i as u8).wrapping_mul(7) }).collect(),
        vec![s, k, n],
    )?;
    insert(exec, "ma_bitset", ma_bitset.clone())?;
    insert(exec, "mb_bitset", mb_bitset.clone())?;
    insert(exec, "mb_bitset_b", mb_bitset_b)?;
    insert(exec, "in_mm_bitset", ma_bitset.clone())?;
    insert(exec, "in_mm_b_bitset", ma_bitset)?;

    Ok(())
}

fn main() -> Result<()> {
    let device = select_device()?;
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../res/models/ops_broadcast_variants_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            a_i8: i8[V];
            b_i8: i8[V];
            b_i8_b: i8[S];
            in_add_i8: i8[V];
            in_add_b_i8: i8[V];
            in_mul_i8: i8[V];
            in_mul_b_i8: i8[V];
            a_i16: i16[V];
            b_i16: i16[V];
            b_i16_b: i16[S];
            in_add_i16: i16[V];
            in_add_b_i16: i16[V];
            in_mul_i16: i16[V];
            in_mul_b_i16: i16[V];
            a_i32: i32[V];
            b_i32: i32[V];
            b_i32_b: i32[S];
            in_add_i32: i32[V];
            in_add_b_i32: i32[V];
            in_mul_i32: i32[V];
            in_mul_b_i32: i32[V];
            a_i64: i64[V];
            b_i64: i64[V];
            b_i64_b: i64[S];
            in_add_i64: i64[V];
            in_add_b_i64: i64[V];
            in_mul_i64: i64[V];
            in_mul_b_i64: i64[V];
            a_u8: u8[V];
            b_u8: u8[V];
            b_u8_b: u8[S];
            in_add_u8: u8[V];
            in_add_b_u8: u8[V];
            in_mul_u8: u8[V];
            in_mul_b_u8: u8[V];
            a_u16: u16[V];
            b_u16: u16[V];
            b_u16_b: u16[S];
            in_add_u16: u16[V];
            in_add_b_u16: u16[V];
            in_mul_u16: u16[V];
            in_mul_b_u16: u16[V];
            a_u32: u32[V];
            b_u32: u32[V];
            b_u32_b: u32[S];
            in_add_u32: u32[V];
            in_add_b_u32: u32[V];
            in_mul_u32: u32[V];
            in_mul_b_u32: u32[V];
            a_u64: u64[V];
            b_u64: u64[V];
            b_u64_b: u64[S];
            in_add_u64: u64[V];
            in_add_b_u64: u64[V];
            in_mul_u64: u64[V];
            in_mul_b_u64: u64[V];
            a_f16: f16[V];
            b_f16: f16[V];
            b_f16_b: f16[S];
            in_add_f16: f16[V];
            in_add_b_f16: f16[V];
            in_mul_f16: f16[V];
            in_mul_b_f16: f16[V];
            a_bf16: bf16[V];
            b_bf16: bf16[V];
            b_bf16_b: bf16[S];
            in_add_bf16: bf16[V];
            in_add_b_bf16: bf16[V];
            in_mul_bf16: bf16[V];
            in_mul_b_bf16: bf16[V];
            a_f8: f8[V];
            b_f8: f8[V];
            b_f8_b: f8[S];
            in_add_f8: f8[V];
            in_add_b_f8: f8[V];
            in_mul_f8: f8[V];
            in_mul_b_f8: f8[V];
            a_f32: f32[V];
            b_f32: f32[V];
            b_f32_b: f32[S];
            in_add_f32: f32[V];
            in_add_b_f32: f32[V];
            in_mul_f32: f32[V];
            in_mul_b_f32: f32[V];
            a_f64: f64[V];
            b_f64: f64[V];
            b_f64_b: f64[S];
            in_add_f64: f64[V];
            in_add_b_f64: f64[V];
            in_mul_f64: f64[V];
            in_mul_b_f64: f64[V];
            a_bool: bool[V];
            b_bool: bool[V];
            b_bool_b: bool[S];
            in_add_bool: bool[V];
            in_add_b_bool: bool[V];
            in_mul_bool: bool[V];
            in_mul_b_bool: bool[V];
            a_bitset: bitset[V];
            b_bitset: bitset[V];
            b_bitset_b: bitset[S];
            in_add_bitset: bitset[V];
            in_add_b_bitset: bitset[V];
            in_mul_bitset: bitset[V];
            in_mul_b_bitset: bitset[V];
            a_i4: i4[V];
            b_i4: i4[V];
            b_i4_b: i4[S];
            in_add_i4: i4[V];
            in_add_b_i4: i4[V];
            in_mul_i4: i4[V];
            in_mul_b_i4: i4[V];
            a_i2: i2[V];
            b_i2: i2[V];
            b_i2_b: i2[S];
            in_add_i2: i2[V];
            in_add_b_i2: i2[V];
            in_mul_i2: i2[V];
            in_mul_b_i2: i2[V];
            a_i1: i1[V];
            b_i1: i1[V];
            b_i1_b: i1[S];
            in_add_i1: i1[V];
            in_add_b_i1: i1[V];
            in_mul_i1: i1[V];
            in_mul_b_i1: i1[V];
            a_u4: u4[V];
            b_u4: u4[V];
            b_u4_b: u4[S];
            in_add_u4: u4[V];
            in_add_b_u4: u4[V];
            in_mul_u4: u4[V];
            in_mul_b_u4: u4[V];
            a_u2: u2[V];
            b_u2: u2[V];
            b_u2_b: u2[S];
            in_add_u2: u2[V];
            in_add_b_u2: u2[V];
            in_mul_u2: u2[V];
            in_mul_b_u2: u2[V];
            a_u1: u1[V];
            b_u1: u1[V];
            b_u1_b: u1[S];
            in_add_u1: u1[V];
            in_add_b_u1: u1[V];
            in_mul_u1: u1[V];
            in_mul_b_u1: u1[V];

            ma_i8: i8[B, M, K];
            mb_i8: i8[B, K, N];
            mb_i8_b: i8[S, K, N];
            in_mm_i8: i8[B, M, K];
            in_mm_b_i8: i8[B, M, K];
            ma_i16: i16[B, M, K];
            mb_i16: i16[B, K, N];
            mb_i16_b: i16[S, K, N];
            in_mm_i16: i16[B, M, K];
            in_mm_b_i16: i16[B, M, K];
            ma_i32: i32[B, M, K];
            mb_i32: i32[B, K, N];
            mb_i32_b: i32[S, K, N];
            in_mm_i32: i32[B, M, K];
            in_mm_b_i32: i32[B, M, K];
            ma_i64: i64[B, M, K];
            mb_i64: i64[B, K, N];
            mb_i64_b: i64[S, K, N];
            in_mm_i64: i64[B, M, K];
            in_mm_b_i64: i64[B, M, K];
            ma_u8: u8[B, M, K];
            mb_u8: u8[B, K, N];
            mb_u8_b: u8[S, K, N];
            in_mm_u8: u8[B, M, K];
            in_mm_b_u8: u8[B, M, K];
            ma_u16: u16[B, M, K];
            mb_u16: u16[B, K, N];
            mb_u16_b: u16[S, K, N];
            in_mm_u16: u16[B, M, K];
            in_mm_b_u16: u16[B, M, K];
            ma_u32: u32[B, M, K];
            mb_u32: u32[B, K, N];
            mb_u32_b: u32[S, K, N];
            in_mm_u32: u32[B, M, K];
            in_mm_b_u32: u32[B, M, K];
            ma_u64: u64[B, M, K];
            mb_u64: u64[B, K, N];
            mb_u64_b: u64[S, K, N];
            in_mm_u64: u64[B, M, K];
            in_mm_b_u64: u64[B, M, K];
            ma_f16: f16[B, M, K];
            mb_f16: f16[B, K, N];
            mb_f16_b: f16[S, K, N];
            in_mm_f16: f16[B, M, K];
            in_mm_b_f16: f16[B, M, K];
            ma_f32: f32[B, M, K];
            mb_f32: f32[B, K, N];
            mb_f32_b: f32[S, K, N];
            in_mm_f32: f32[B, M, K];
            in_mm_b_f32: f32[B, M, K];
            ma_f64: f64[B, M, K];
            mb_f64: f64[B, K, N];
            mb_f64_b: f64[S, K, N];
            in_mm_f64: f64[B, M, K];
            in_mm_b_f64: f64[B, M, K];
            ma_bool: bool[B, M, K];
            mb_bool: bool[B, K, N];
            mb_bool_b: bool[S, K, N];
            in_mm_bool: bool[B, M, K];
            in_mm_b_bool: bool[B, M, K];
            ma_bitset: bitset[B, M, K];
            mb_bitset: bitset[B, K, N];
            mb_bitset_b: bitset[S, K, N];
            in_mm_bitset: bitset[B, M, K];
            in_mm_b_bitset: bitset[B, M, K];
        }

        volatile {
            add_i8: i8[V];
            add_b_i8: i8[V];
            mul_i8: i8[V];
            mul_b_i8: i8[V];
            add_i16: i16[V];
            add_b_i16: i16[V];
            mul_i16: i16[V];
            mul_b_i16: i16[V];
            add_i32: i32[V];
            add_b_i32: i32[V];
            mul_i32: i32[V];
            mul_b_i32: i32[V];
            add_i64: i64[V];
            add_b_i64: i64[V];
            mul_i64: i64[V];
            mul_b_i64: i64[V];
            add_u8: u8[V];
            add_b_u8: u8[V];
            mul_u8: u8[V];
            mul_b_u8: u8[V];
            add_u16: u16[V];
            add_b_u16: u16[V];
            mul_u16: u16[V];
            mul_b_u16: u16[V];
            add_u32: u32[V];
            add_b_u32: u32[V];
            mul_u32: u32[V];
            mul_b_u32: u32[V];
            add_u64: u64[V];
            add_b_u64: u64[V];
            mul_u64: u64[V];
            mul_b_u64: u64[V];
            add_f16: f16[V];
            add_b_f16: f16[V];
            mul_f16: f16[V];
            mul_b_f16: f16[V];
            add_bf16: bf16[V];
            add_b_bf16: bf16[V];
            mul_bf16: bf16[V];
            mul_b_bf16: bf16[V];
            add_f8: f8[V];
            add_b_f8: f8[V];
            mul_f8: f8[V];
            mul_b_f8: f8[V];
            add_f32: f32[V];
            add_b_f32: f32[V];
            mul_f32: f32[V];
            mul_b_f32: f32[V];
            add_f64: f64[V];
            add_b_f64: f64[V];
            mul_f64: f64[V];
            mul_b_f64: f64[V];
            add_bool: bool[V];
            add_b_bool: bool[V];
            mul_bool: bool[V];
            mul_b_bool: bool[V];
            add_bitset: bitset[V];
            add_b_bitset: bitset[V];
            mul_bitset: bitset[V];
            mul_b_bitset: bitset[V];
            add_i4: i4[V];
            add_b_i4: i4[V];
            mul_i4: i4[V];
            mul_b_i4: i4[V];
            add_i2: i2[V];
            add_b_i2: i2[V];
            mul_i2: i2[V];
            mul_b_i2: i2[V];
            add_i1: i1[V];
            add_b_i1: i1[V];
            mul_i1: i1[V];
            mul_b_i1: i1[V];
            add_u4: u4[V];
            add_b_u4: u4[V];
            mul_u4: u4[V];
            mul_b_u4: u4[V];
            add_u2: u2[V];
            add_b_u2: u2[V];
            mul_u2: u2[V];
            mul_b_u2: u2[V];
            add_u1: u1[V];
            add_b_u1: u1[V];
            mul_u1: u1[V];
            mul_b_u1: u1[V];

            mm_i8: i8[B, M, N];
            mm_b_i8: i8[B, M, N];
            mm_i16: i16[B, M, N];
            mm_b_i16: i16[B, M, N];
            mm_i32: i32[B, M, N];
            mm_b_i32: i32[B, M, N];
            mm_i64: i64[B, M, N];
            mm_b_i64: i64[B, M, N];
            mm_u8: u8[B, M, N];
            mm_b_u8: u8[B, M, N];
            mm_u16: u16[B, M, N];
            mm_b_u16: u16[B, M, N];
            mm_u32: u32[B, M, N];
            mm_b_u32: u32[B, M, N];
            mm_u64: u64[B, M, N];
            mm_b_u64: u64[B, M, N];
            mm_f16: f16[B, M, N];
            mm_b_f16: f16[B, M, N];
            mm_f32: f32[B, M, N];
            mm_b_f32: f32[B, M, N];
            mm_f64: f64[B, M, N];
            mm_b_f64: f64[B, M, N];
            mm_bool: bool[B, M, N];
            mm_b_bool: bool[B, M, N];
            mm_bitset: bitset[B, M, N];
            mm_b_bitset: bitset[B, M, N];

            add_acc_i8: i16[V];
            add_acc_b_i8: i16[V];
            add_acc_i16: i32[V];
            add_acc_b_i16: i32[V];
            add_acc_i32: i64[V];
            add_acc_b_i32: i64[V];
            add_acc_u8: u16[V];
            add_acc_b_u8: u16[V];
            add_acc_u16: u32[V];
            add_acc_b_u16: u32[V];
            add_acc_u32: u64[V];
            add_acc_b_u32: u64[V];
            add_acc_i4: i8[V];
            add_acc_b_i4: i8[V];
            add_acc_i2: i8[V];
            add_acc_b_i2: i8[V];
            add_acc_i1: i8[V];
            add_acc_b_i1: i8[V];
            add_acc_u4: u8[V];
            add_acc_b_u4: u8[V];
            add_acc_u2: u8[V];
            add_acc_b_u2: u8[V];
            add_acc_u1: u8[V];
            add_acc_b_u1: u8[V];

            mul_acc_i8: i16[V];
            mul_acc_b_i8: i16[V];
            mul_acc_i16: i32[V];
            mul_acc_b_i16: i32[V];
            mul_acc_i32: i64[V];
            mul_acc_b_i32: i64[V];
            mul_acc_u8: u16[V];
            mul_acc_b_u8: u16[V];
            mul_acc_u16: u32[V];
            mul_acc_b_u16: u32[V];
            mul_acc_u32: u64[V];
            mul_acc_b_u32: u64[V];
            mul_acc_i4: i8[V];
            mul_acc_b_i4: i8[V];
            mul_acc_i2: i8[V];
            mul_acc_b_i2: i8[V];
            mul_acc_i1: i8[V];
            mul_acc_b_i1: i8[V];
            mul_acc_u4: u8[V];
            mul_acc_b_u4: u8[V];
            mul_acc_u2: u8[V];
            mul_acc_b_u2: u8[V];
            mul_acc_u1: u8[V];
            mul_acc_b_u1: u8[V];

            mm_acc_i8: i16[B, M, N];
            mm_acc_b_i8: i16[B, M, N];
            mm_acc_i16: i32[B, M, N];
            mm_acc_b_i16: i32[B, M, N];
            mm_acc_i32: i64[B, M, N];
            mm_acc_b_i32: i64[B, M, N];
            mm_acc_u8: u16[B, M, N];
            mm_acc_b_u8: u16[B, M, N];
            mm_acc_u16: u32[B, M, N];
            mm_acc_b_u16: u32[B, M, N];
            mm_acc_u32: u64[B, M, N];
            mm_acc_b_u32: u64[B, M, N];
        }

        block entry {
            op add(a_i8, b_i8) >> add_i8;
            op add(a_i8, b_i8_b) >> add_b_i8;
            op mul(a_i8, b_i8) >> mul_i8;
            op mul(a_i8, b_i8_b) >> mul_b_i8;
            op add(in_add_i8, b_i8) >> in_add_i8;
            op add(in_add_b_i8, b_i8_b) >> in_add_b_i8;
            op mul(in_mul_i8, b_i8) >> in_mul_i8;
            op mul(in_mul_b_i8, b_i8_b) >> in_mul_b_i8;
            op add(a_i16, b_i16) >> add_i16;
            op add(a_i16, b_i16_b) >> add_b_i16;
            op mul(a_i16, b_i16) >> mul_i16;
            op mul(a_i16, b_i16_b) >> mul_b_i16;
            op add(in_add_i16, b_i16) >> in_add_i16;
            op add(in_add_b_i16, b_i16_b) >> in_add_b_i16;
            op mul(in_mul_i16, b_i16) >> in_mul_i16;
            op mul(in_mul_b_i16, b_i16_b) >> in_mul_b_i16;
            op add(a_i32, b_i32) >> add_i32;
            op add(a_i32, b_i32_b) >> add_b_i32;
            op mul(a_i32, b_i32) >> mul_i32;
            op mul(a_i32, b_i32_b) >> mul_b_i32;
            op add(in_add_i32, b_i32) >> in_add_i32;
            op add(in_add_b_i32, b_i32_b) >> in_add_b_i32;
            op mul(in_mul_i32, b_i32) >> in_mul_i32;
            op mul(in_mul_b_i32, b_i32_b) >> in_mul_b_i32;
            op add(a_i64, b_i64) >> add_i64;
            op add(a_i64, b_i64_b) >> add_b_i64;
            op mul(a_i64, b_i64) >> mul_i64;
            op mul(a_i64, b_i64_b) >> mul_b_i64;
            op add(in_add_i64, b_i64) >> in_add_i64;
            op add(in_add_b_i64, b_i64_b) >> in_add_b_i64;
            op mul(in_mul_i64, b_i64) >> in_mul_i64;
            op mul(in_mul_b_i64, b_i64_b) >> in_mul_b_i64;
            op add(a_u8, b_u8) >> add_u8;
            op add(a_u8, b_u8_b) >> add_b_u8;
            op mul(a_u8, b_u8) >> mul_u8;
            op mul(a_u8, b_u8_b) >> mul_b_u8;
            op add(in_add_u8, b_u8) >> in_add_u8;
            op add(in_add_b_u8, b_u8_b) >> in_add_b_u8;
            op mul(in_mul_u8, b_u8) >> in_mul_u8;
            op mul(in_mul_b_u8, b_u8_b) >> in_mul_b_u8;
            op add(a_u16, b_u16) >> add_u16;
            op add(a_u16, b_u16_b) >> add_b_u16;
            op mul(a_u16, b_u16) >> mul_u16;
            op mul(a_u16, b_u16_b) >> mul_b_u16;
            op add(in_add_u16, b_u16) >> in_add_u16;
            op add(in_add_b_u16, b_u16_b) >> in_add_b_u16;
            op mul(in_mul_u16, b_u16) >> in_mul_u16;
            op mul(in_mul_b_u16, b_u16_b) >> in_mul_b_u16;
            op add(a_u32, b_u32) >> add_u32;
            op add(a_u32, b_u32_b) >> add_b_u32;
            op mul(a_u32, b_u32) >> mul_u32;
            op mul(a_u32, b_u32_b) >> mul_b_u32;
            op add(in_add_u32, b_u32) >> in_add_u32;
            op add(in_add_b_u32, b_u32_b) >> in_add_b_u32;
            op mul(in_mul_u32, b_u32) >> in_mul_u32;
            op mul(in_mul_b_u32, b_u32_b) >> in_mul_b_u32;
            op add(a_u64, b_u64) >> add_u64;
            op add(a_u64, b_u64_b) >> add_b_u64;
            op mul(a_u64, b_u64) >> mul_u64;
            op mul(a_u64, b_u64_b) >> mul_b_u64;
            op add(in_add_u64, b_u64) >> in_add_u64;
            op add(in_add_b_u64, b_u64_b) >> in_add_b_u64;
            op mul(in_mul_u64, b_u64) >> in_mul_u64;
            op mul(in_mul_b_u64, b_u64_b) >> in_mul_b_u64;
            op add(a_f16, b_f16) >> add_f16;
            op add(a_f16, b_f16_b) >> add_b_f16;
            op mul(a_f16, b_f16) >> mul_f16;
            op mul(a_f16, b_f16_b) >> mul_b_f16;
            op add(in_add_f16, b_f16) >> in_add_f16;
            op add(in_add_b_f16, b_f16_b) >> in_add_b_f16;
            op mul(in_mul_f16, b_f16) >> in_mul_f16;
            op mul(in_mul_b_f16, b_f16_b) >> in_mul_b_f16;
            op add(a_bf16, b_bf16) >> add_bf16;
            op add(a_bf16, b_bf16_b) >> add_b_bf16;
            op mul(a_bf16, b_bf16) >> mul_bf16;
            op mul(a_bf16, b_bf16_b) >> mul_b_bf16;
            op add(in_add_bf16, b_bf16) >> in_add_bf16;
            op add(in_add_b_bf16, b_bf16_b) >> in_add_b_bf16;
            op mul(in_mul_bf16, b_bf16) >> in_mul_bf16;
            op mul(in_mul_b_bf16, b_bf16_b) >> in_mul_b_bf16;
            op add(a_f8, b_f8) >> add_f8;
            op add(a_f8, b_f8_b) >> add_b_f8;
            op mul(a_f8, b_f8) >> mul_f8;
            op mul(a_f8, b_f8_b) >> mul_b_f8;
            op add(in_add_f8, b_f8) >> in_add_f8;
            op add(in_add_b_f8, b_f8_b) >> in_add_b_f8;
            op mul(in_mul_f8, b_f8) >> in_mul_f8;
            op mul(in_mul_b_f8, b_f8_b) >> in_mul_b_f8;
            op add(a_f32, b_f32) >> add_f32;
            op add(a_f32, b_f32_b) >> add_b_f32;
            op mul(a_f32, b_f32) >> mul_f32;
            op mul(a_f32, b_f32_b) >> mul_b_f32;
            op add(in_add_f32, b_f32) >> in_add_f32;
            op add(in_add_b_f32, b_f32_b) >> in_add_b_f32;
            op mul(in_mul_f32, b_f32) >> in_mul_f32;
            op mul(in_mul_b_f32, b_f32_b) >> in_mul_b_f32;
            op add(a_f64, b_f64) >> add_f64;
            op add(a_f64, b_f64_b) >> add_b_f64;
            op mul(a_f64, b_f64) >> mul_f64;
            op mul(a_f64, b_f64_b) >> mul_b_f64;
            op add(in_add_f64, b_f64) >> in_add_f64;
            op add(in_add_b_f64, b_f64_b) >> in_add_b_f64;
            op mul(in_mul_f64, b_f64) >> in_mul_f64;
            op mul(in_mul_b_f64, b_f64_b) >> in_mul_b_f64;
            op add(a_bool, b_bool) >> add_bool;
            op add(a_bool, b_bool_b) >> add_b_bool;
            op mul(a_bool, b_bool) >> mul_bool;
            op mul(a_bool, b_bool_b) >> mul_b_bool;
            op add(in_add_bool, b_bool) >> in_add_bool;
            op add(in_add_b_bool, b_bool_b) >> in_add_b_bool;
            op mul(in_mul_bool, b_bool) >> in_mul_bool;
            op mul(in_mul_b_bool, b_bool_b) >> in_mul_b_bool;
            op add(a_bitset, b_bitset) >> add_bitset;
            op add(a_bitset, b_bitset_b) >> add_b_bitset;
            op mul(a_bitset, b_bitset) >> mul_bitset;
            op mul(a_bitset, b_bitset_b) >> mul_b_bitset;
            op add(in_add_bitset, b_bitset) >> in_add_bitset;
            op add(in_add_b_bitset, b_bitset_b) >> in_add_b_bitset;
            op mul(in_mul_bitset, b_bitset) >> in_mul_bitset;
            op mul(in_mul_b_bitset, b_bitset_b) >> in_mul_b_bitset;
            op add(a_i4, b_i4) >> add_i4;
            op add(a_i4, b_i4_b) >> add_b_i4;
            op mul(a_i4, b_i4) >> mul_i4;
            op mul(a_i4, b_i4_b) >> mul_b_i4;
            op add(in_add_i4, b_i4) >> in_add_i4;
            op add(in_add_b_i4, b_i4_b) >> in_add_b_i4;
            op mul(in_mul_i4, b_i4) >> in_mul_i4;
            op mul(in_mul_b_i4, b_i4_b) >> in_mul_b_i4;
            op add(a_i2, b_i2) >> add_i2;
            op add(a_i2, b_i2_b) >> add_b_i2;
            op mul(a_i2, b_i2) >> mul_i2;
            op mul(a_i2, b_i2_b) >> mul_b_i2;
            op add(in_add_i2, b_i2) >> in_add_i2;
            op add(in_add_b_i2, b_i2_b) >> in_add_b_i2;
            op mul(in_mul_i2, b_i2) >> in_mul_i2;
            op mul(in_mul_b_i2, b_i2_b) >> in_mul_b_i2;
            op add(a_i1, b_i1) >> add_i1;
            op add(a_i1, b_i1_b) >> add_b_i1;
            op mul(a_i1, b_i1) >> mul_i1;
            op mul(a_i1, b_i1_b) >> mul_b_i1;
            op add(in_add_i1, b_i1) >> in_add_i1;
            op add(in_add_b_i1, b_i1_b) >> in_add_b_i1;
            op mul(in_mul_i1, b_i1) >> in_mul_i1;
            op mul(in_mul_b_i1, b_i1_b) >> in_mul_b_i1;
            op add(a_u4, b_u4) >> add_u4;
            op add(a_u4, b_u4_b) >> add_b_u4;
            op mul(a_u4, b_u4) >> mul_u4;
            op mul(a_u4, b_u4_b) >> mul_b_u4;
            op add(in_add_u4, b_u4) >> in_add_u4;
            op add(in_add_b_u4, b_u4_b) >> in_add_b_u4;
            op mul(in_mul_u4, b_u4) >> in_mul_u4;
            op mul(in_mul_b_u4, b_u4_b) >> in_mul_b_u4;
            op add(a_u2, b_u2) >> add_u2;
            op add(a_u2, b_u2_b) >> add_b_u2;
            op mul(a_u2, b_u2) >> mul_u2;
            op mul(a_u2, b_u2_b) >> mul_b_u2;
            op add(in_add_u2, b_u2) >> in_add_u2;
            op add(in_add_b_u2, b_u2_b) >> in_add_b_u2;
            op mul(in_mul_u2, b_u2) >> in_mul_u2;
            op mul(in_mul_b_u2, b_u2_b) >> in_mul_b_u2;
            op add(a_u1, b_u1) >> add_u1;
            op add(a_u1, b_u1_b) >> add_b_u1;
            op mul(a_u1, b_u1) >> mul_u1;
            op mul(a_u1, b_u1_b) >> mul_b_u1;
            op add(in_add_u1, b_u1) >> in_add_u1;
            op add(in_add_b_u1, b_u1_b) >> in_add_b_u1;
            op mul(in_mul_u1, b_u1) >> in_mul_u1;
            op mul(in_mul_b_u1, b_u1_b) >> in_mul_b_u1;

            op matmul(ma_i8, mb_i8) >> mm_i8;
            op matmul(ma_i8, mb_i8_b) >> mm_b_i8;
            op matmul(in_mm_i8, mb_i8) >> in_mm_i8;
            op matmul(in_mm_b_i8, mb_i8_b) >> in_mm_b_i8;
            op matmul(ma_i16, mb_i16) >> mm_i16;
            op matmul(ma_i16, mb_i16_b) >> mm_b_i16;
            op matmul(in_mm_i16, mb_i16) >> in_mm_i16;
            op matmul(in_mm_b_i16, mb_i16_b) >> in_mm_b_i16;
            op matmul(ma_i32, mb_i32) >> mm_i32;
            op matmul(ma_i32, mb_i32_b) >> mm_b_i32;
            op matmul(in_mm_i32, mb_i32) >> in_mm_i32;
            op matmul(in_mm_b_i32, mb_i32_b) >> in_mm_b_i32;
            op matmul(ma_i64, mb_i64) >> mm_i64;
            op matmul(ma_i64, mb_i64_b) >> mm_b_i64;
            op matmul(in_mm_i64, mb_i64) >> in_mm_i64;
            op matmul(in_mm_b_i64, mb_i64_b) >> in_mm_b_i64;
            op matmul(ma_u8, mb_u8) >> mm_u8;
            op matmul(ma_u8, mb_u8_b) >> mm_b_u8;
            op matmul(in_mm_u8, mb_u8) >> in_mm_u8;
            op matmul(in_mm_b_u8, mb_u8_b) >> in_mm_b_u8;
            op matmul(ma_u16, mb_u16) >> mm_u16;
            op matmul(ma_u16, mb_u16_b) >> mm_b_u16;
            op matmul(in_mm_u16, mb_u16) >> in_mm_u16;
            op matmul(in_mm_b_u16, mb_u16_b) >> in_mm_b_u16;
            op matmul(ma_u32, mb_u32) >> mm_u32;
            op matmul(ma_u32, mb_u32_b) >> mm_b_u32;
            op matmul(in_mm_u32, mb_u32) >> in_mm_u32;
            op matmul(in_mm_b_u32, mb_u32_b) >> in_mm_b_u32;
            op matmul(ma_u64, mb_u64) >> mm_u64;
            op matmul(ma_u64, mb_u64_b) >> mm_b_u64;
            op matmul(in_mm_u64, mb_u64) >> in_mm_u64;
            op matmul(in_mm_b_u64, mb_u64_b) >> in_mm_b_u64;
            op matmul(ma_f16, mb_f16) >> mm_f16;
            op matmul(ma_f16, mb_f16_b) >> mm_b_f16;
            op matmul(in_mm_f16, mb_f16) >> in_mm_f16;
            op matmul(in_mm_b_f16, mb_f16_b) >> in_mm_b_f16;
            op matmul(ma_f32, mb_f32) >> mm_f32;
            op matmul(ma_f32, mb_f32_b) >> mm_b_f32;
            op matmul(in_mm_f32, mb_f32) >> in_mm_f32;
            op matmul(in_mm_b_f32, mb_f32_b) >> in_mm_b_f32;
            op matmul(ma_f64, mb_f64) >> mm_f64;
            op matmul(ma_f64, mb_f64_b) >> mm_b_f64;
            op matmul(in_mm_f64, mb_f64) >> in_mm_f64;
            op matmul(in_mm_b_f64, mb_f64_b) >> in_mm_b_f64;
            op matmul(ma_bool, mb_bool) >> mm_bool;
            op matmul(ma_bool, mb_bool_b) >> mm_b_bool;
            op matmul(in_mm_bool, mb_bool) >> in_mm_bool;
            op matmul(in_mm_b_bool, mb_bool_b) >> in_mm_b_bool;
            op matmul(ma_bitset, mb_bitset) >> mm_bitset;
            op matmul(ma_bitset, mb_bitset_b) >> mm_b_bitset;
            op matmul(in_mm_bitset, mb_bitset) >> in_mm_bitset;
            op matmul(in_mm_b_bitset, mb_bitset_b) >> in_mm_b_bitset;

            op add(a_i8, b_i8, acc=i16) >> add_acc_i8;
            op add(a_i8, b_i8_b, acc=i16) >> add_acc_b_i8;
            op add(a_i16, b_i16, acc=i32) >> add_acc_i16;
            op add(a_i16, b_i16_b, acc=i32) >> add_acc_b_i16;
            op add(a_i32, b_i32, acc=i64) >> add_acc_i32;
            op add(a_i32, b_i32_b, acc=i64) >> add_acc_b_i32;
            op add(a_u8, b_u8, acc=u16) >> add_acc_u8;
            op add(a_u8, b_u8_b, acc=u16) >> add_acc_b_u8;
            op add(a_u16, b_u16, acc=u32) >> add_acc_u16;
            op add(a_u16, b_u16_b, acc=u32) >> add_acc_b_u16;
            op add(a_u32, b_u32, acc=u64) >> add_acc_u32;
            op add(a_u32, b_u32_b, acc=u64) >> add_acc_b_u32;
            op add(a_i4, b_i4, acc=i8) >> add_acc_i4;
            op add(a_i4, b_i4_b, acc=i8) >> add_acc_b_i4;
            op add(a_i2, b_i2, acc=i8) >> add_acc_i2;
            op add(a_i2, b_i2_b, acc=i8) >> add_acc_b_i2;
            op add(a_i1, b_i1, acc=i8) >> add_acc_i1;
            op add(a_i1, b_i1_b, acc=i8) >> add_acc_b_i1;
            op add(a_u4, b_u4, acc=u8) >> add_acc_u4;
            op add(a_u4, b_u4_b, acc=u8) >> add_acc_b_u4;
            op add(a_u2, b_u2, acc=u8) >> add_acc_u2;
            op add(a_u2, b_u2_b, acc=u8) >> add_acc_b_u2;
            op add(a_u1, b_u1, acc=u8) >> add_acc_u1;
            op add(a_u1, b_u1_b, acc=u8) >> add_acc_b_u1;

            op mul(a_i8, b_i8, acc=i16) >> mul_acc_i8;
            op mul(a_i8, b_i8_b, acc=i16) >> mul_acc_b_i8;
            op mul(a_i16, b_i16, acc=i32) >> mul_acc_i16;
            op mul(a_i16, b_i16_b, acc=i32) >> mul_acc_b_i16;
            op mul(a_i32, b_i32, acc=i64) >> mul_acc_i32;
            op mul(a_i32, b_i32_b, acc=i64) >> mul_acc_b_i32;
            op mul(a_u8, b_u8, acc=u16) >> mul_acc_u8;
            op mul(a_u8, b_u8_b, acc=u16) >> mul_acc_b_u8;
            op mul(a_u16, b_u16, acc=u32) >> mul_acc_u16;
            op mul(a_u16, b_u16_b, acc=u32) >> mul_acc_b_u16;
            op mul(a_u32, b_u32, acc=u64) >> mul_acc_u32;
            op mul(a_u32, b_u32_b, acc=u64) >> mul_acc_b_u32;
            op mul(a_i4, b_i4, acc=i8) >> mul_acc_i4;
            op mul(a_i4, b_i4_b, acc=i8) >> mul_acc_b_i4;
            op mul(a_i2, b_i2, acc=i8) >> mul_acc_i2;
            op mul(a_i2, b_i2_b, acc=i8) >> mul_acc_b_i2;
            op mul(a_i1, b_i1, acc=i8) >> mul_acc_i1;
            op mul(a_i1, b_i1_b, acc=i8) >> mul_acc_b_i1;
            op mul(a_u4, b_u4, acc=u8) >> mul_acc_u4;
            op mul(a_u4, b_u4_b, acc=u8) >> mul_acc_b_u4;
            op mul(a_u2, b_u2, acc=u8) >> mul_acc_u2;
            op mul(a_u2, b_u2_b, acc=u8) >> mul_acc_b_u2;
            op mul(a_u1, b_u1, acc=u8) >> mul_acc_u1;
            op mul(a_u1, b_u1_b, acc=u8) >> mul_acc_b_u1;

            op matmul(ma_i8, mb_i8, acc=i16) >> mm_acc_i8;
            op matmul(ma_i8, mb_i8_b, acc=i16) >> mm_acc_b_i8;
            op matmul(ma_i16, mb_i16, acc=i32) >> mm_acc_i16;
            op matmul(ma_i16, mb_i16_b, acc=i32) >> mm_acc_b_i16;
            op matmul(ma_i32, mb_i32, acc=i64) >> mm_acc_i32;
            op matmul(ma_i32, mb_i32_b, acc=i64) >> mm_acc_b_i32;
            op matmul(ma_u8, mb_u8, acc=u16) >> mm_acc_u8;
            op matmul(ma_u8, mb_u8_b, acc=u16) >> mm_acc_b_u8;
            op matmul(ma_u16, mb_u16, acc=u32) >> mm_acc_u16;
            op matmul(ma_u16, mb_u16_b, acc=u32) >> mm_acc_b_u16;
            op matmul(ma_u32, mb_u32, acc=u64) >> mm_acc_u32;
            op matmul(ma_u32, mb_u32_b, acc=u64) >> mm_acc_b_u32;
            return;
        }
    };

    let v = model.size_of("V")?;
    let s = model.size_of("S")?;
    let m = model.size_of("M")?;
    let k = model.size_of("K")?;
    let n = model.size_of("N")?;
    let b = model.size_of("B")?;

    let sim_cpu = Simulator::new(&model, &g, Device::Cpu)?.with_inplace();
    let mut exec_cpu = sim_cpu.make_executor()?;
    populate_exec(&mut exec_cpu, v, s, m, k, n, b)?;
    exec_cpu.step()?;
    let mut refs = HashMap::new();
    let mut float_refs = HashMap::new();

    collect_named::<i8>(&mut refs, &mut exec_cpu, I8_OUTPUTS)?;
    collect_named::<i16>(&mut refs, &mut exec_cpu, I16_OUTPUTS)?;
    collect_named::<i32>(&mut refs, &mut exec_cpu, I32_OUTPUTS)?;
    collect_named::<i64>(&mut refs, &mut exec_cpu, I64_OUTPUTS)?;
    collect_named::<u8>(&mut refs, &mut exec_cpu, U8_OUTPUTS)?;
    collect_named::<u16>(&mut refs, &mut exec_cpu, U16_OUTPUTS)?;
    collect_named::<u32>(&mut refs, &mut exec_cpu, U32_OUTPUTS)?;
    collect_named::<u64>(&mut refs, &mut exec_cpu, U64_OUTPUTS)?;
    collect_named::<bool>(&mut refs, &mut exec_cpu, BOOL_OUTPUTS)?;
    collect_named::<Bitset>(&mut refs, &mut exec_cpu, BITSET_OUTPUTS)?;
    collect_named::<I4>(&mut refs, &mut exec_cpu, I4_OUTPUTS)?;
    collect_named::<I2>(&mut refs, &mut exec_cpu, I2_OUTPUTS)?;
    collect_named::<I1>(&mut refs, &mut exec_cpu, I1_OUTPUTS)?;
    collect_named::<U4>(&mut refs, &mut exec_cpu, U4_OUTPUTS)?;
    collect_named::<U2>(&mut refs, &mut exec_cpu, U2_OUTPUTS)?;
    collect_named::<U1>(&mut refs, &mut exec_cpu, U1_OUTPUTS)?;
    collect_named_float::<F16>(&mut refs, &mut float_refs, &mut exec_cpu, F16_OUTPUTS)?;
    collect_named_float::<BF16>(&mut refs, &mut float_refs, &mut exec_cpu, BF16_OUTPUTS)?;
    collect_named_float::<F8E5M2>(&mut refs, &mut float_refs, &mut exec_cpu, F8_OUTPUTS)?;
    collect_named_float::<f32>(&mut refs, &mut float_refs, &mut exec_cpu, F32_OUTPUTS)?;
    collect_named_float::<f64>(&mut refs, &mut float_refs, &mut exec_cpu, F64_OUTPUTS)?;

    let sim = Simulator::new(&model, &g, device)?.with_inplace();
    let mut exec = sim.make_executor()?;
    populate_exec(&mut exec, v, s, m, k, n, b)?;
    exec.step()?;

    log::info!("⚠️ == Pass but drift. ✅ == Pass with no drift. ❌ == Fail");

    validate_named::<i8>(&refs, &mut exec, I8_OUTPUTS)?;
    validate_named::<i16>(&refs, &mut exec, I16_OUTPUTS)?;
    validate_named::<i32>(&refs, &mut exec, I32_OUTPUTS)?;
    validate_named::<i64>(&refs, &mut exec, I64_OUTPUTS)?;
    validate_named::<u8>(&refs, &mut exec, U8_OUTPUTS)?;
    validate_named::<u16>(&refs, &mut exec, U16_OUTPUTS)?;
    validate_named::<u32>(&refs, &mut exec, U32_OUTPUTS)?;
    validate_named::<u64>(&refs, &mut exec, U64_OUTPUTS)?;
    validate_named::<bool>(&refs, &mut exec, BOOL_OUTPUTS)?;
    validate_named::<Bitset>(&refs, &mut exec, BITSET_OUTPUTS)?;
    validate_named::<I4>(&refs, &mut exec, I4_OUTPUTS)?;
    validate_named::<I2>(&refs, &mut exec, I2_OUTPUTS)?;
    validate_named::<I1>(&refs, &mut exec, I1_OUTPUTS)?;
    validate_named::<U4>(&refs, &mut exec, U4_OUTPUTS)?;
    validate_named::<U2>(&refs, &mut exec, U2_OUTPUTS)?;
    validate_named::<U1>(&refs, &mut exec, U1_OUTPUTS)?;
    validate_named_float::<F16>(&refs, &float_refs, &mut exec, F16_OUTPUTS, FloatTol::f16())?;
    validate_named_float::<BF16>(&refs, &float_refs, &mut exec, BF16_OUTPUTS, FloatTol::bf16())?;
    validate_named_float::<F8E5M2>(&refs, &float_refs, &mut exec, F8_OUTPUTS, FloatTol::f8())?;
    validate_named_float::<f32>(&refs, &float_refs, &mut exec, F32_OUTPUTS, FloatTol::f32())?;
    validate_named_float::<f64>(&refs, &float_refs, &mut exec, F64_OUTPUTS, FloatTol::f64())?;

    log::info!("ops_broadcast_variants completed on {:?}", device);

    Ok(())
}
