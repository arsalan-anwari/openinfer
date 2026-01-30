use openinfer::{
    graph, Bitset, Device, Executor, F16, Fetchable, BF16, F8, I1, I2, I4, ModelLoader, Simulator,
    Tensor, TensorElement, TensorOptions, U1, U2, U4,
};
use openinfer::{format_truncated, FormatValue};
use std::collections::HashMap;
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

fn tensor_packed_i4(values: &[i8], shape: Vec<usize>) -> anyhow::Result<Tensor<I4>> {
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

fn tensor_packed_i2(values: &[i8], shape: Vec<usize>) -> anyhow::Result<Tensor<I2>> {
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

fn tensor_packed_i1(values: &[i8], shape: Vec<usize>) -> anyhow::Result<Tensor<I1>> {
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

fn tensor_packed_u4(values: &[u8], shape: Vec<usize>) -> anyhow::Result<Tensor<U4>> {
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

fn tensor_packed_u2(values: &[u8], shape: Vec<usize>) -> anyhow::Result<Tensor<U2>> {
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

fn tensor_packed_u1(values: &[u8], shape: Vec<usize>) -> anyhow::Result<Tensor<U1>> {
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

fn insert<T: TensorElement>(exec: &mut Executor, name: &str, tensor: Tensor<T>) -> anyhow::Result<()> {
    exec.insert_dynamic(name, <T as TensorElement>::into_value(tensor))?;
    Ok(())
}

fn format_tensor<T: TensorElement + FormatValue>(
    exec: &mut Executor,
    name: &str,
) -> anyhow::Result<String> {
    let tensor: Tensor<T> = exec.fetch(name)?;
    let limit = 10.min(tensor.data.len());
    Ok(format!("{}", format_truncated(&tensor.data[..limit])))
}

fn format_scalar<T: FormatValue + Copy>(value: T) -> String {
    format!("{}", format_truncated(std::slice::from_ref(&value)))
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

impl ToF64 for F8 {
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
        Self {
            abs: 0.6,
            rel: 0.08,
        }
    }

    fn bf16() -> Self {
        Self {
            abs: 0.1,
            rel: 0.02,
        }
    }

    fn f8() -> Self {
        Self {
            abs: 0.6,
            rel: 0.25,
        }
    }

    fn f32() -> Self {
        Self {
            abs: 1e-4,
            rel: 1e-4,
        }
    }

    fn f64() -> Self {
        Self {
            abs: 1e-8,
            rel: 1e-8,
        }
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
) -> anyhow::Result<()> {
    let formatted = format_tensor::<T>(exec, name)?;
    refs.insert(name.to_string(), formatted);
    Ok(())
}

fn collect_tensor_ref_float<T: TensorElement + FormatValue + Copy + ToF64>(
    refs: &mut HashMap<String, String>,
    float_refs: &mut HashMap<String, Vec<f64>>,
    exec: &mut Executor,
    name: &str,
) -> anyhow::Result<()> {
    collect_tensor_ref::<T>(refs, exec, name)?;
    let tensor: Tensor<T> = exec.fetch(name)?;
    let values = tensor.data.iter().map(|v| v.to_f64()).collect();
    float_refs.insert(name.to_string(), values);
    Ok(())
}

fn collect_scalar_ref<T: FormatValue + Copy + Fetchable>(
    refs: &mut HashMap<String, String>,
    exec: &mut Executor,
    name: &str,
) -> anyhow::Result<()> {
    let value: T = exec.fetch(name)?;
    refs.insert(name.to_string(), format_scalar(value));
    Ok(())
}

fn validate_tensor<T: TensorElement + FormatValue>(
    refs: &HashMap<String, String>,
    exec: &mut Executor,
    name: &str,
) -> anyhow::Result<()> {
    let formatted = format_tensor::<T>(exec, name)?;
    let ref_val = refs.get(name).cloned().unwrap_or_else(|| "<missing>".to_string());
    let status = if formatted == ref_val { "✅" } else { "❌" };
    openinfer::log!("[{}] {} = {} -- ref = {}", status, name, formatted, ref_val);
    Ok(())
}

fn validate_tensor_float<T: TensorElement + FormatValue + Copy + ToF64>(
    refs: &HashMap<String, String>,
    float_refs: &HashMap<String, Vec<f64>>,
    exec: &mut Executor,
    name: &str,
    tol: FloatTol,
) -> anyhow::Result<()> {
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
    openinfer::log!("[{}] {} = {} -- ref = {}", status, name, formatted, ref_val);
    Ok(())
}

fn validate_scalar<T: FormatValue + Copy + Fetchable>(
    refs: &HashMap<String, String>,
    exec: &mut Executor,
    name: &str,
) -> anyhow::Result<()> {
    let value: T = exec.fetch(name)?;
    let formatted = format_scalar(value);
    let ref_val = refs.get(name).cloned().unwrap_or_else(|| "<missing>".to_string());
    let status = if formatted == ref_val { "✅" } else { "❌" };
    openinfer::log!("[{}] {} = {} -- ref = {}", status, name, formatted, ref_val);
    Ok(())
}

macro_rules! for_each_tensor_output_collect {
    ($non_float:ident, $float:ident, $exec:expr, $refs:expr, $float_refs:expr) => {
        $non_float::<i8>($refs, $exec, "add_i8")?;
        $non_float::<i16>($refs, $exec, "add_i16")?;
        $non_float::<i32>($refs, $exec, "add_i32")?;
        $non_float::<i64>($refs, $exec, "add_i64")?;
        $non_float::<u8>($refs, $exec, "add_u8")?;
        $non_float::<u16>($refs, $exec, "add_u16")?;
        $non_float::<u32>($refs, $exec, "add_u32")?;
        $non_float::<u64>($refs, $exec, "add_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "add_f16")?;
        $float::<BF16>($refs, $float_refs, $exec, "add_bf16")?;
        $float::<F8>($refs, $float_refs, $exec, "add_f8")?;
        $float::<f32>($refs, $float_refs, $exec, "add_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "add_f64")?;
        $non_float::<bool>($refs, $exec, "add_bool")?;
        $non_float::<Bitset>($refs, $exec, "add_bitset")?;
        $non_float::<I4>($refs, $exec, "add_i4")?;
        $non_float::<I2>($refs, $exec, "add_i2")?;
        $non_float::<I1>($refs, $exec, "add_i1")?;
        $non_float::<U4>($refs, $exec, "add_u4")?;
        $non_float::<U2>($refs, $exec, "add_u2")?;
        $non_float::<U1>($refs, $exec, "add_u1")?;

        $non_float::<i8>($refs, $exec, "mul_i8")?;
        $non_float::<i16>($refs, $exec, "mul_i16")?;
        $non_float::<i32>($refs, $exec, "mul_i32")?;
        $non_float::<i64>($refs, $exec, "mul_i64")?;
        $non_float::<u8>($refs, $exec, "mul_u8")?;
        $non_float::<u16>($refs, $exec, "mul_u16")?;
        $non_float::<u32>($refs, $exec, "mul_u32")?;
        $non_float::<u64>($refs, $exec, "mul_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "mul_f16")?;
        $float::<BF16>($refs, $float_refs, $exec, "mul_bf16")?;
        $float::<F8>($refs, $float_refs, $exec, "mul_f8")?;
        $float::<f32>($refs, $float_refs, $exec, "mul_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "mul_f64")?;
        $non_float::<bool>($refs, $exec, "mul_bool")?;
        $non_float::<Bitset>($refs, $exec, "mul_bitset")?;
        $non_float::<I4>($refs, $exec, "mul_i4")?;
        $non_float::<I2>($refs, $exec, "mul_i2")?;
        $non_float::<I1>($refs, $exec, "mul_i1")?;
        $non_float::<U4>($refs, $exec, "mul_u4")?;
        $non_float::<U2>($refs, $exec, "mul_u2")?;
        $non_float::<U1>($refs, $exec, "mul_u1")?;

        $non_float::<i8>($refs, $exec, "abs_i8")?;
        $non_float::<i16>($refs, $exec, "abs_i16")?;
        $non_float::<i32>($refs, $exec, "abs_i32")?;
        $non_float::<i64>($refs, $exec, "abs_i64")?;
        $float::<F16>($refs, $float_refs, $exec, "abs_f16")?;
        $float::<BF16>($refs, $float_refs, $exec, "abs_bf16")?;
        $float::<F8>($refs, $float_refs, $exec, "abs_f8")?;
        $float::<f32>($refs, $float_refs, $exec, "abs_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "abs_f64")?;
        $non_float::<I4>($refs, $exec, "abs_i4")?;
        $non_float::<I2>($refs, $exec, "abs_i2")?;
        $non_float::<I1>($refs, $exec, "abs_i1")?;

        $non_float::<i8>($refs, $exec, "relu_i8")?;
        $non_float::<i16>($refs, $exec, "relu_i16")?;
        $non_float::<i32>($refs, $exec, "relu_i32")?;
        $non_float::<i64>($refs, $exec, "relu_i64")?;
        $float::<F16>($refs, $float_refs, $exec, "relu_f16")?;
        $float::<BF16>($refs, $float_refs, $exec, "relu_bf16")?;
        $float::<F8>($refs, $float_refs, $exec, "relu_f8")?;
        $float::<f32>($refs, $float_refs, $exec, "relu_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "relu_f64")?;
        $non_float::<I4>($refs, $exec, "relu_i4")?;

        $non_float::<i8>($refs, $exec, "fill_i8")?;
        $non_float::<i16>($refs, $exec, "fill_i16")?;
        $non_float::<i32>($refs, $exec, "fill_i32")?;
        $non_float::<i64>($refs, $exec, "fill_i64")?;
        $non_float::<u8>($refs, $exec, "fill_u8")?;
        $non_float::<u16>($refs, $exec, "fill_u16")?;
        $non_float::<u32>($refs, $exec, "fill_u32")?;
        $non_float::<u64>($refs, $exec, "fill_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "fill_f16")?;
        $float::<BF16>($refs, $float_refs, $exec, "fill_bf16")?;
        $float::<F8>($refs, $float_refs, $exec, "fill_f8")?;
        $float::<f32>($refs, $float_refs, $exec, "fill_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "fill_f64")?;
        $non_float::<bool>($refs, $exec, "fill_bool")?;
        $non_float::<Bitset>($refs, $exec, "fill_bitset")?;
        $non_float::<I4>($refs, $exec, "fill_i4")?;
        $non_float::<I2>($refs, $exec, "fill_i2")?;
        $non_float::<I1>($refs, $exec, "fill_i1")?;
        $non_float::<U4>($refs, $exec, "fill_u4")?;
        $non_float::<U2>($refs, $exec, "fill_u2")?;
        $non_float::<U1>($refs, $exec, "fill_u1")?;

        $non_float::<i8>($refs, $exec, "mm_i8")?;
        $non_float::<i16>($refs, $exec, "mm_i16")?;
        $non_float::<i32>($refs, $exec, "mm_i32")?;
        $non_float::<i64>($refs, $exec, "mm_i64")?;
        $non_float::<u8>($refs, $exec, "mm_u8")?;
        $non_float::<u16>($refs, $exec, "mm_u16")?;
        $non_float::<u32>($refs, $exec, "mm_u32")?;
        $non_float::<u64>($refs, $exec, "mm_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "mm_f16")?;
        $float::<f32>($refs, $float_refs, $exec, "mm_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "mm_f64")?;
        $non_float::<bool>($refs, $exec, "mm_bool")?;
        $non_float::<Bitset>($refs, $exec, "mm_bitset")?;

        $non_float::<i16>($refs, $exec, "add_acc_i8")?;
        $non_float::<i32>($refs, $exec, "add_acc_i16")?;
        $non_float::<i64>($refs, $exec, "add_acc_i32")?;
        $non_float::<u16>($refs, $exec, "add_acc_u8")?;
        $non_float::<u32>($refs, $exec, "add_acc_u16")?;
        $non_float::<u64>($refs, $exec, "add_acc_u32")?;
        $non_float::<i8>($refs, $exec, "add_acc_i4")?;
        $non_float::<i8>($refs, $exec, "add_acc_i2")?;
        $non_float::<i8>($refs, $exec, "add_acc_i1")?;
        $non_float::<u8>($refs, $exec, "add_acc_u4")?;
        $non_float::<u8>($refs, $exec, "add_acc_u2")?;
        $non_float::<u8>($refs, $exec, "add_acc_u1")?;

        $non_float::<i16>($refs, $exec, "mul_acc_i8")?;
        $non_float::<i32>($refs, $exec, "mul_acc_i16")?;
        $non_float::<i64>($refs, $exec, "mul_acc_i32")?;
        $non_float::<u16>($refs, $exec, "mul_acc_u8")?;
        $non_float::<u32>($refs, $exec, "mul_acc_u16")?;
        $non_float::<u64>($refs, $exec, "mul_acc_u32")?;
        $non_float::<i8>($refs, $exec, "mul_acc_i4")?;
        $non_float::<i8>($refs, $exec, "mul_acc_i2")?;
        $non_float::<i8>($refs, $exec, "mul_acc_i1")?;
        $non_float::<u8>($refs, $exec, "mul_acc_u4")?;
        $non_float::<u8>($refs, $exec, "mul_acc_u2")?;
        $non_float::<u8>($refs, $exec, "mul_acc_u1")?;

        $non_float::<i16>($refs, $exec, "abs_acc_i8")?;
        $non_float::<i32>($refs, $exec, "abs_acc_i16")?;
        $non_float::<i64>($refs, $exec, "abs_acc_i32")?;
        $non_float::<i8>($refs, $exec, "abs_acc_i4")?;
        $non_float::<i8>($refs, $exec, "abs_acc_i2")?;
        $non_float::<i8>($refs, $exec, "abs_acc_i1")?;

        $non_float::<i16>($refs, $exec, "mm_acc_i8")?;
        $non_float::<i32>($refs, $exec, "mm_acc_i16")?;
        $non_float::<i64>($refs, $exec, "mm_acc_i32")?;
        $non_float::<u16>($refs, $exec, "mm_acc_u8")?;
        $non_float::<u32>($refs, $exec, "mm_acc_u16")?;
        $non_float::<u64>($refs, $exec, "mm_acc_u32")?;

        $non_float::<i8>($refs, $exec, "in_i8")?;
        $non_float::<i16>($refs, $exec, "in_i16")?;
        $non_float::<i32>($refs, $exec, "in_i32")?;
        $non_float::<i64>($refs, $exec, "in_i64")?;
        $non_float::<u8>($refs, $exec, "in_u8")?;
        $non_float::<u16>($refs, $exec, "in_u16")?;
        $non_float::<u32>($refs, $exec, "in_u32")?;
        $non_float::<u64>($refs, $exec, "in_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "in_f16")?;
        $float::<BF16>($refs, $float_refs, $exec, "in_bf16")?;
        $float::<F8>($refs, $float_refs, $exec, "in_f8")?;
        $float::<f32>($refs, $float_refs, $exec, "in_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "in_f64")?;
        $non_float::<bool>($refs, $exec, "in_bool")?;
        $non_float::<Bitset>($refs, $exec, "in_bitset")?;
        $non_float::<I4>($refs, $exec, "in_i4")?;
        $non_float::<I2>($refs, $exec, "in_i2")?;
        $non_float::<I1>($refs, $exec, "in_i1")?;
        $non_float::<U4>($refs, $exec, "in_u4")?;
        $non_float::<U2>($refs, $exec, "in_u2")?;
        $non_float::<U1>($refs, $exec, "in_u1")?;

        $non_float::<i8>($refs, $exec, "in_ma_i8")?;
        $non_float::<i16>($refs, $exec, "in_ma_i16")?;
        $non_float::<i32>($refs, $exec, "in_ma_i32")?;
        $non_float::<i64>($refs, $exec, "in_ma_i64")?;
        $non_float::<u8>($refs, $exec, "in_ma_u8")?;
        $non_float::<u16>($refs, $exec, "in_ma_u16")?;
        $non_float::<u32>($refs, $exec, "in_ma_u32")?;
        $non_float::<u64>($refs, $exec, "in_ma_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "in_ma_f16")?;
        $float::<f32>($refs, $float_refs, $exec, "in_ma_f32")?;
        $float::<f64>($refs, $float_refs, $exec, "in_ma_f64")?;
        $non_float::<bool>($refs, $exec, "in_ma_bool")?;
        $non_float::<Bitset>($refs, $exec, "in_ma_bitset")?;
    };
}

macro_rules! for_each_tensor_output_validate {
    ($non_float:ident, $float:ident, $exec:expr, $refs:expr, $float_refs:expr) => {
        $non_float::<i8>($refs, $exec, "add_i8")?;
        $non_float::<i16>($refs, $exec, "add_i16")?;
        $non_float::<i32>($refs, $exec, "add_i32")?;
        $non_float::<i64>($refs, $exec, "add_i64")?;
        $non_float::<u8>($refs, $exec, "add_u8")?;
        $non_float::<u16>($refs, $exec, "add_u16")?;
        $non_float::<u32>($refs, $exec, "add_u32")?;
        $non_float::<u64>($refs, $exec, "add_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "add_f16", FloatTol::f16())?;
        $float::<BF16>($refs, $float_refs, $exec, "add_bf16", FloatTol::bf16())?;
        $float::<F8>($refs, $float_refs, $exec, "add_f8", FloatTol::f8())?;
        $float::<f32>($refs, $float_refs, $exec, "add_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "add_f64", FloatTol::f64())?;
        $non_float::<bool>($refs, $exec, "add_bool")?;
        $non_float::<Bitset>($refs, $exec, "add_bitset")?;
        $non_float::<I4>($refs, $exec, "add_i4")?;
        $non_float::<I2>($refs, $exec, "add_i2")?;
        $non_float::<I1>($refs, $exec, "add_i1")?;
        $non_float::<U4>($refs, $exec, "add_u4")?;
        $non_float::<U2>($refs, $exec, "add_u2")?;
        $non_float::<U1>($refs, $exec, "add_u1")?;

        $non_float::<i8>($refs, $exec, "mul_i8")?;
        $non_float::<i16>($refs, $exec, "mul_i16")?;
        $non_float::<i32>($refs, $exec, "mul_i32")?;
        $non_float::<i64>($refs, $exec, "mul_i64")?;
        $non_float::<u8>($refs, $exec, "mul_u8")?;
        $non_float::<u16>($refs, $exec, "mul_u16")?;
        $non_float::<u32>($refs, $exec, "mul_u32")?;
        $non_float::<u64>($refs, $exec, "mul_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "mul_f16", FloatTol::f16())?;
        $float::<BF16>($refs, $float_refs, $exec, "mul_bf16", FloatTol::bf16())?;
        $float::<F8>($refs, $float_refs, $exec, "mul_f8", FloatTol::f8())?;
        $float::<f32>($refs, $float_refs, $exec, "mul_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "mul_f64", FloatTol::f64())?;
        $non_float::<bool>($refs, $exec, "mul_bool")?;
        $non_float::<Bitset>($refs, $exec, "mul_bitset")?;
        $non_float::<I4>($refs, $exec, "mul_i4")?;
        $non_float::<I2>($refs, $exec, "mul_i2")?;
        $non_float::<I1>($refs, $exec, "mul_i1")?;
        $non_float::<U4>($refs, $exec, "mul_u4")?;
        $non_float::<U2>($refs, $exec, "mul_u2")?;
        $non_float::<U1>($refs, $exec, "mul_u1")?;

        $non_float::<i8>($refs, $exec, "abs_i8")?;
        $non_float::<i16>($refs, $exec, "abs_i16")?;
        $non_float::<i32>($refs, $exec, "abs_i32")?;
        $non_float::<i64>($refs, $exec, "abs_i64")?;
        $float::<F16>($refs, $float_refs, $exec, "abs_f16", FloatTol::f16())?;
        $float::<BF16>($refs, $float_refs, $exec, "abs_bf16", FloatTol::bf16())?;
        $float::<F8>($refs, $float_refs, $exec, "abs_f8", FloatTol::f8())?;
        $float::<f32>($refs, $float_refs, $exec, "abs_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "abs_f64", FloatTol::f64())?;
        $non_float::<I4>($refs, $exec, "abs_i4")?;
        $non_float::<I2>($refs, $exec, "abs_i2")?;
        $non_float::<I1>($refs, $exec, "abs_i1")?;

        $non_float::<i8>($refs, $exec, "relu_i8")?;
        $non_float::<i16>($refs, $exec, "relu_i16")?;
        $non_float::<i32>($refs, $exec, "relu_i32")?;
        $non_float::<i64>($refs, $exec, "relu_i64")?;
        $float::<F16>($refs, $float_refs, $exec, "relu_f16", FloatTol::f16())?;
        $float::<BF16>($refs, $float_refs, $exec, "relu_bf16", FloatTol::bf16())?;
        $float::<F8>($refs, $float_refs, $exec, "relu_f8", FloatTol::f8())?;
        $float::<f32>($refs, $float_refs, $exec, "relu_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "relu_f64", FloatTol::f64())?;
        $non_float::<I4>($refs, $exec, "relu_i4")?;

        $non_float::<i8>($refs, $exec, "fill_i8")?;
        $non_float::<i16>($refs, $exec, "fill_i16")?;
        $non_float::<i32>($refs, $exec, "fill_i32")?;
        $non_float::<i64>($refs, $exec, "fill_i64")?;
        $non_float::<u8>($refs, $exec, "fill_u8")?;
        $non_float::<u16>($refs, $exec, "fill_u16")?;
        $non_float::<u32>($refs, $exec, "fill_u32")?;
        $non_float::<u64>($refs, $exec, "fill_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "fill_f16", FloatTol::f16())?;
        $float::<BF16>($refs, $float_refs, $exec, "fill_bf16", FloatTol::bf16())?;
        $float::<F8>($refs, $float_refs, $exec, "fill_f8", FloatTol::f8())?;
        $float::<f32>($refs, $float_refs, $exec, "fill_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "fill_f64", FloatTol::f64())?;
        $non_float::<bool>($refs, $exec, "fill_bool")?;
        $non_float::<Bitset>($refs, $exec, "fill_bitset")?;
        $non_float::<I4>($refs, $exec, "fill_i4")?;
        $non_float::<I2>($refs, $exec, "fill_i2")?;
        $non_float::<I1>($refs, $exec, "fill_i1")?;
        $non_float::<U4>($refs, $exec, "fill_u4")?;
        $non_float::<U2>($refs, $exec, "fill_u2")?;
        $non_float::<U1>($refs, $exec, "fill_u1")?;

        $non_float::<i8>($refs, $exec, "mm_i8")?;
        $non_float::<i16>($refs, $exec, "mm_i16")?;
        $non_float::<i32>($refs, $exec, "mm_i32")?;
        $non_float::<i64>($refs, $exec, "mm_i64")?;
        $non_float::<u8>($refs, $exec, "mm_u8")?;
        $non_float::<u16>($refs, $exec, "mm_u16")?;
        $non_float::<u32>($refs, $exec, "mm_u32")?;
        $non_float::<u64>($refs, $exec, "mm_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "mm_f16", FloatTol::f16())?;
        $float::<f32>($refs, $float_refs, $exec, "mm_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "mm_f64", FloatTol::f64())?;
        $non_float::<bool>($refs, $exec, "mm_bool")?;
        $non_float::<Bitset>($refs, $exec, "mm_bitset")?;

        $non_float::<i16>($refs, $exec, "add_acc_i8")?;
        $non_float::<i32>($refs, $exec, "add_acc_i16")?;
        $non_float::<i64>($refs, $exec, "add_acc_i32")?;
        $non_float::<u16>($refs, $exec, "add_acc_u8")?;
        $non_float::<u32>($refs, $exec, "add_acc_u16")?;
        $non_float::<u64>($refs, $exec, "add_acc_u32")?;
        $non_float::<i8>($refs, $exec, "add_acc_i4")?;
        $non_float::<i8>($refs, $exec, "add_acc_i2")?;
        $non_float::<i8>($refs, $exec, "add_acc_i1")?;
        $non_float::<u8>($refs, $exec, "add_acc_u4")?;
        $non_float::<u8>($refs, $exec, "add_acc_u2")?;
        $non_float::<u8>($refs, $exec, "add_acc_u1")?;

        $non_float::<i16>($refs, $exec, "mul_acc_i8")?;
        $non_float::<i32>($refs, $exec, "mul_acc_i16")?;
        $non_float::<i64>($refs, $exec, "mul_acc_i32")?;
        $non_float::<u16>($refs, $exec, "mul_acc_u8")?;
        $non_float::<u32>($refs, $exec, "mul_acc_u16")?;
        $non_float::<u64>($refs, $exec, "mul_acc_u32")?;
        $non_float::<i8>($refs, $exec, "mul_acc_i4")?;
        $non_float::<i8>($refs, $exec, "mul_acc_i2")?;
        $non_float::<i8>($refs, $exec, "mul_acc_i1")?;
        $non_float::<u8>($refs, $exec, "mul_acc_u4")?;
        $non_float::<u8>($refs, $exec, "mul_acc_u2")?;
        $non_float::<u8>($refs, $exec, "mul_acc_u1")?;

        $non_float::<i16>($refs, $exec, "abs_acc_i8")?;
        $non_float::<i32>($refs, $exec, "abs_acc_i16")?;
        $non_float::<i64>($refs, $exec, "abs_acc_i32")?;
        $non_float::<i8>($refs, $exec, "abs_acc_i4")?;
        $non_float::<i8>($refs, $exec, "abs_acc_i2")?;
        $non_float::<i8>($refs, $exec, "abs_acc_i1")?;

        $non_float::<i16>($refs, $exec, "mm_acc_i8")?;
        $non_float::<i32>($refs, $exec, "mm_acc_i16")?;
        $non_float::<i64>($refs, $exec, "mm_acc_i32")?;
        $non_float::<u16>($refs, $exec, "mm_acc_u8")?;
        $non_float::<u32>($refs, $exec, "mm_acc_u16")?;
        $non_float::<u64>($refs, $exec, "mm_acc_u32")?;

        $non_float::<i8>($refs, $exec, "in_i8")?;
        $non_float::<i16>($refs, $exec, "in_i16")?;
        $non_float::<i32>($refs, $exec, "in_i32")?;
        $non_float::<i64>($refs, $exec, "in_i64")?;
        $non_float::<u8>($refs, $exec, "in_u8")?;
        $non_float::<u16>($refs, $exec, "in_u16")?;
        $non_float::<u32>($refs, $exec, "in_u32")?;
        $non_float::<u64>($refs, $exec, "in_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "in_f16", FloatTol::f16())?;
        $float::<BF16>($refs, $float_refs, $exec, "in_bf16", FloatTol::bf16())?;
        $float::<F8>($refs, $float_refs, $exec, "in_f8", FloatTol::f8())?;
        $float::<f32>($refs, $float_refs, $exec, "in_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "in_f64", FloatTol::f64())?;
        $non_float::<bool>($refs, $exec, "in_bool")?;
        $non_float::<Bitset>($refs, $exec, "in_bitset")?;
        $non_float::<I4>($refs, $exec, "in_i4")?;
        $non_float::<I2>($refs, $exec, "in_i2")?;
        $non_float::<I1>($refs, $exec, "in_i1")?;
        $non_float::<U4>($refs, $exec, "in_u4")?;
        $non_float::<U2>($refs, $exec, "in_u2")?;
        $non_float::<U1>($refs, $exec, "in_u1")?;

        $non_float::<i8>($refs, $exec, "in_ma_i8")?;
        $non_float::<i16>($refs, $exec, "in_ma_i16")?;
        $non_float::<i32>($refs, $exec, "in_ma_i32")?;
        $non_float::<i64>($refs, $exec, "in_ma_i64")?;
        $non_float::<u8>($refs, $exec, "in_ma_u8")?;
        $non_float::<u16>($refs, $exec, "in_ma_u16")?;
        $non_float::<u32>($refs, $exec, "in_ma_u32")?;
        $non_float::<u64>($refs, $exec, "in_ma_u64")?;
        $float::<F16>($refs, $float_refs, $exec, "in_ma_f16", FloatTol::f16())?;
        $float::<f32>($refs, $float_refs, $exec, "in_ma_f32", FloatTol::f32())?;
        $float::<f64>($refs, $float_refs, $exec, "in_ma_f64", FloatTol::f64())?;
        $non_float::<bool>($refs, $exec, "in_ma_bool")?;
        $non_float::<Bitset>($refs, $exec, "in_ma_bitset")?;
    };
}

macro_rules! for_each_scalar_output {
    ($func:ident, $exec:expr, $refs:expr) => {
        $func::<bool>($refs, $exec, "finite_f16")?;
        $func::<bool>($refs, $exec, "finite_bf16")?;
        $func::<bool>($refs, $exec, "finite_f8")?;
        $func::<bool>($refs, $exec, "finite_f32")?;
        $func::<bool>($refs, $exec, "finite_f64")?;
    };
}

fn populate_exec(
    exec: &mut Executor,
    v: usize,
    m: usize,
    k: usize,
    n: usize,
    b: usize,
) -> anyhow::Result<()> {
    let i8_vals: Vec<i8> = (0..v).map(|i| i as i8 - 4).collect();
    let i8_vals_b: Vec<i8> = (0..v).map(|i| i as i8 - 1).collect();
    let u8_vals: Vec<u8> = (0..v).map(|i| i as u8).collect();
    let u8_vals_b: Vec<u8> = (0..v).map(|i| (i as u8).wrapping_add(1)).collect();

    insert(exec, "a_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;
    insert(exec, "b_i8", tensor_with_shape(i8_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_i8", tensor_with_shape(i8_vals.clone(), vec![v])?)?;
    insert(exec, "a_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "b_i16", tensor_with_shape(i8_vals_b.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "in_i16", tensor_with_shape(i8_vals.iter().map(|v| *v as i16).collect(), vec![v])?)?;
    insert(exec, "a_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "b_i32", tensor_with_shape(i8_vals_b.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "in_i32", tensor_with_shape(i8_vals.iter().map(|v| *v as i32).collect(), vec![v])?)?;
    insert(exec, "a_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "b_i64", tensor_with_shape(i8_vals_b.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "in_i64", tensor_with_shape(i8_vals.iter().map(|v| *v as i64).collect(), vec![v])?)?;
    insert(exec, "a_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;
    insert(exec, "b_u8", tensor_with_shape(u8_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_u8", tensor_with_shape(u8_vals.clone(), vec![v])?)?;
    insert(exec, "a_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "b_u16", tensor_with_shape(u8_vals_b.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "in_u16", tensor_with_shape(u8_vals.iter().map(|v| *v as u16).collect(), vec![v])?)?;
    insert(exec, "a_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "b_u32", tensor_with_shape(u8_vals_b.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "in_u32", tensor_with_shape(u8_vals.iter().map(|v| *v as u32).collect(), vec![v])?)?;
    insert(exec, "a_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "b_u64", tensor_with_shape(u8_vals_b.iter().map(|v| *v as u64).collect(), vec![v])?)?;
    insert(exec, "in_u64", tensor_with_shape(u8_vals.iter().map(|v| *v as u64).collect(), vec![v])?)?;

    let f16_vals: Vec<F16> = (0..v).map(|i| F16::from_f32(i as f32 * 0.5 - 2.0)).collect();
    let f16_vals_b: Vec<F16> = (0..v).map(|i| F16::from_f32(i as f32 * 0.5 + 0.5)).collect();
    let bf16_vals: Vec<BF16> = (0..v).map(|i| BF16::from_f32(i as f32 * 0.5 - 2.0)).collect();
    let bf16_vals_b: Vec<BF16> = (0..v).map(|i| BF16::from_f32(i as f32 * 0.5 + 0.5)).collect();
    let f8_vals: Vec<F8> = (0..v).map(|i| F8::from_f32(i as f32 * 0.5 - 2.0)).collect();
    let f8_vals_b: Vec<F8> = (0..v).map(|i| F8::from_f32(i as f32 * 0.5 + 0.5)).collect();
    let f32_vals: Vec<f32> = (0..v).map(|i| i as f32 * 0.5 - 2.0).collect();
    let f32_vals_b: Vec<f32> = (0..v).map(|i| i as f32 * 0.5 + 0.5).collect();
    let f64_vals: Vec<f64> = (0..v).map(|i| i as f64 * 0.5 - 2.0).collect();
    let f64_vals_b: Vec<f64> = (0..v).map(|i| i as f64 * 0.5 + 0.5).collect();

    insert(exec, "a_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;
    insert(exec, "b_f16", tensor_with_shape(f16_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_f16", tensor_with_shape(f16_vals.clone(), vec![v])?)?;
    insert(exec, "a_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;
    insert(exec, "b_bf16", tensor_with_shape(bf16_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_bf16", tensor_with_shape(bf16_vals.clone(), vec![v])?)?;
    insert(exec, "a_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;
    insert(exec, "b_f8", tensor_with_shape(f8_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_f8", tensor_with_shape(f8_vals.clone(), vec![v])?)?;
    insert(exec, "a_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;
    insert(exec, "b_f32", tensor_with_shape(f32_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_f32", tensor_with_shape(f32_vals.clone(), vec![v])?)?;
    insert(exec, "a_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;
    insert(exec, "b_f64", tensor_with_shape(f64_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_f64", tensor_with_shape(f64_vals.clone(), vec![v])?)?;

    let bool_vals: Vec<bool> = (0..v).map(|i| i % 2 == 0).collect();
    let bool_vals_b: Vec<bool> = (0..v).map(|i| i % 3 == 0).collect();
    insert(exec, "a_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;
    insert(exec, "b_bool", tensor_with_shape(bool_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_bool", tensor_with_shape(bool_vals.clone(), vec![v])?)?;

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
    insert(exec, "a_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;
    insert(exec, "b_bitset", tensor_with_shape(bitset_vals_b.clone(), vec![v])?)?;
    insert(exec, "in_bitset", tensor_with_shape(bitset_vals.clone(), vec![v])?)?;

    let i4_vals = vec![-8, -4, -1, 0, 1, 3, 6, 7];
    let i4_vals_b = vec![1, -1, 2, -2, 3, -3, 4, -4];
    insert(exec, "a_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;
    insert(exec, "b_i4", tensor_packed_i4(&i4_vals_b, vec![v])?)?;
    insert(exec, "in_i4", tensor_packed_i4(&i4_vals, vec![v])?)?;
    let i2_vals = vec![-2, -1, 0, 1, -2, -1, 0, 1];
    let i2_vals_b = vec![1, 0, -1, -2, 1, 0, -1, -2];
    insert(exec, "a_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;
    insert(exec, "b_i2", tensor_packed_i2(&i2_vals_b, vec![v])?)?;
    insert(exec, "in_i2", tensor_packed_i2(&i2_vals, vec![v])?)?;
    let i1_vals = vec![-1, 0, -1, 0, -1, 0, -1, 0];
    let i1_vals_b = vec![0, -1, 0, -1, 0, -1, 0, -1];
    insert(exec, "a_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;
    insert(exec, "b_i1", tensor_packed_i1(&i1_vals_b, vec![v])?)?;
    insert(exec, "in_i1", tensor_packed_i1(&i1_vals, vec![v])?)?;

    let u4_vals = vec![0, 1, 2, 3, 4, 5, 14, 15];
    let u4_vals_b = vec![1, 2, 3, 4, 0, 1, 2, 3];
    insert(exec, "a_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;
    insert(exec, "b_u4", tensor_packed_u4(&u4_vals_b, vec![v])?)?;
    insert(exec, "in_u4", tensor_packed_u4(&u4_vals, vec![v])?)?;
    let u2_vals = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let u2_vals_b = vec![3, 2, 1, 0, 3, 2, 1, 0];
    insert(exec, "a_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;
    insert(exec, "b_u2", tensor_packed_u2(&u2_vals_b, vec![v])?)?;
    insert(exec, "in_u2", tensor_packed_u2(&u2_vals, vec![v])?)?;
    let u1_vals = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let u1_vals_b = vec![1, 0, 1, 0, 0, 1, 0, 1];
    insert(exec, "a_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;
    insert(exec, "b_u1", tensor_packed_u1(&u1_vals_b, vec![v])?)?;
    insert(exec, "in_u1", tensor_packed_u1(&u1_vals, vec![v])?)?;

    let mk = m * k;
    let kn = k * n;
    let batch_mk = b * mk;
    let batch_kn = b * kn;

    let ma_i8 = tensor_with_shape((0..batch_mk).map(|i| i as i8 - 2).collect(), vec![b, m, k])?;
    let mb_i8 = tensor_with_shape((0..batch_kn).map(|i| i as i8 - 1).collect(), vec![b, k, n])?;
    insert(exec, "ma_i8", ma_i8.clone())?;
    insert(exec, "mb_i8", mb_i8.clone())?;
    insert(exec, "in_ma_i8", ma_i8)?;

    let ma_i16 = tensor_with_shape((0..batch_mk).map(|i| i as i16 - 2).collect(), vec![b, m, k])?;
    let mb_i16 = tensor_with_shape((0..batch_kn).map(|i| i as i16 - 1).collect(), vec![b, k, n])?;
    insert(exec, "ma_i16", ma_i16.clone())?;
    insert(exec, "mb_i16", mb_i16.clone())?;
    insert(exec, "in_ma_i16", ma_i16)?;

    let ma_i32 = tensor_with_shape((0..batch_mk).map(|i| i as i32 - 2).collect(), vec![b, m, k])?;
    let mb_i32 = tensor_with_shape((0..batch_kn).map(|i| i as i32 - 1).collect(), vec![b, k, n])?;
    insert(exec, "ma_i32", ma_i32.clone())?;
    insert(exec, "mb_i32", mb_i32.clone())?;
    insert(exec, "in_ma_i32", ma_i32)?;

    let ma_i64 = tensor_with_shape((0..batch_mk).map(|i| i as i64 - 2).collect(), vec![b, m, k])?;
    let mb_i64 = tensor_with_shape((0..batch_kn).map(|i| i as i64 - 1).collect(), vec![b, k, n])?;
    insert(exec, "ma_i64", ma_i64.clone())?;
    insert(exec, "mb_i64", mb_i64.clone())?;
    insert(exec, "in_ma_i64", ma_i64)?;

    let ma_u8 = tensor_with_shape((0..batch_mk).map(|i| i as u8).collect(), vec![b, m, k])?;
    let mb_u8 = tensor_with_shape((0..batch_kn).map(|i| (i as u8).wrapping_add(1)).collect(), vec![b, k, n])?;
    insert(exec, "ma_u8", ma_u8.clone())?;
    insert(exec, "mb_u8", mb_u8.clone())?;
    insert(exec, "in_ma_u8", ma_u8)?;

    let ma_u16 = tensor_with_shape((0..batch_mk).map(|i| i as u16).collect(), vec![b, m, k])?;
    let mb_u16 = tensor_with_shape((0..batch_kn).map(|i| (i as u16).wrapping_add(1)).collect(), vec![b, k, n])?;
    insert(exec, "ma_u16", ma_u16.clone())?;
    insert(exec, "mb_u16", mb_u16.clone())?;
    insert(exec, "in_ma_u16", ma_u16)?;

    let ma_u32 = tensor_with_shape((0..batch_mk).map(|i| i as u32).collect(), vec![b, m, k])?;
    let mb_u32 = tensor_with_shape((0..batch_kn).map(|i| (i as u32).wrapping_add(1)).collect(), vec![b, k, n])?;
    insert(exec, "ma_u32", ma_u32.clone())?;
    insert(exec, "mb_u32", mb_u32.clone())?;
    insert(exec, "in_ma_u32", ma_u32)?;

    let ma_u64 = tensor_with_shape((0..batch_mk).map(|i| i as u64).collect(), vec![b, m, k])?;
    let mb_u64 = tensor_with_shape((0..batch_kn).map(|i| (i as u64).wrapping_add(1)).collect(), vec![b, k, n])?;
    insert(exec, "ma_u64", ma_u64.clone())?;
    insert(exec, "mb_u64", mb_u64.clone())?;
    insert(exec, "in_ma_u64", ma_u64)?;

    let ma_f16 = tensor_with_shape(
        (0..batch_mk).map(|i| F16::from_f32(i as f32 * 0.25 - 1.0)).collect(),
        vec![b, m, k],
    )?;
    let mb_f16 = tensor_with_shape(
        (0..batch_kn).map(|i| F16::from_f32(i as f32 * 0.25 + 0.5)).collect(),
        vec![b, k, n],
    )?;
    insert(exec, "ma_f16", ma_f16.clone())?;
    insert(exec, "mb_f16", mb_f16.clone())?;
    insert(exec, "in_ma_f16", ma_f16)?;

    let ma_f32 = tensor_with_shape(
        (0..batch_mk).map(|i| i as f32 * 0.25 - 1.0).collect(),
        vec![b, m, k],
    )?;
    let mb_f32 = tensor_with_shape(
        (0..batch_kn).map(|i| i as f32 * 0.25 + 0.5).collect(),
        vec![b, k, n],
    )?;
    insert(exec, "ma_f32", ma_f32.clone())?;
    insert(exec, "mb_f32", mb_f32.clone())?;
    insert(exec, "in_ma_f32", ma_f32)?;

    let ma_f64 = tensor_with_shape(
        (0..batch_mk).map(|i| i as f64 * 0.25 - 1.0).collect(),
        vec![b, m, k],
    )?;
    let mb_f64 = tensor_with_shape(
        (0..batch_kn).map(|i| i as f64 * 0.25 + 0.5).collect(),
        vec![b, k, n],
    )?;
    insert(exec, "ma_f64", ma_f64.clone())?;
    insert(exec, "mb_f64", mb_f64.clone())?;
    insert(exec, "in_ma_f64", ma_f64)?;

    let ma_bool = tensor_with_shape((0..batch_mk).map(|i| i % 2 == 0).collect(), vec![b, m, k])?;
    let mb_bool = tensor_with_shape((0..batch_kn).map(|i| i % 3 == 0).collect(), vec![b, k, n])?;
    insert(exec, "ma_bool", ma_bool.clone())?;
    insert(exec, "mb_bool", mb_bool.clone())?;
    insert(exec, "in_ma_bool", ma_bool)?;

    let ma_bitset = tensor_with_shape(
        (0..batch_mk).map(|i| Bitset { bits: (i as u8).wrapping_mul(2) }).collect(),
        vec![b, m, k],
    )?;
    let mb_bitset = tensor_with_shape(
        (0..batch_kn).map(|i| Bitset { bits: (i as u8).wrapping_mul(7) }).collect(),
        vec![b, k, n],
    )?;
    insert(exec, "ma_bitset", ma_bitset.clone())?;
    insert(exec, "mb_bitset", mb_bitset.clone())?;
    insert(exec, "in_ma_bitset", ma_bitset)?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = select_device()?;
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/ops_accumulate_inplace_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            a_i8: i8[V];
            b_i8: i8[V];
            in_i8: i8[V];
            a_i16: i16[V];
            b_i16: i16[V];
            in_i16: i16[V];
            a_i32: i32[V];
            b_i32: i32[V];
            in_i32: i32[V];
            a_i64: i64[V];
            b_i64: i64[V];
            in_i64: i64[V];
            a_u8: u8[V];
            b_u8: u8[V];
            in_u8: u8[V];
            a_u16: u16[V];
            b_u16: u16[V];
            in_u16: u16[V];
            a_u32: u32[V];
            b_u32: u32[V];
            in_u32: u32[V];
            a_u64: u64[V];
            b_u64: u64[V];
            in_u64: u64[V];
            a_f16: f16[V];
            b_f16: f16[V];
            in_f16: f16[V];
            a_bf16: bf16[V];
            b_bf16: bf16[V];
            in_bf16: bf16[V];
            a_f8: f8[V];
            b_f8: f8[V];
            in_f8: f8[V];
            a_f32: f32[V];
            b_f32: f32[V];
            in_f32: f32[V];
            a_f64: f64[V];
            b_f64: f64[V];
            in_f64: f64[V];
            a_bool: bool[V];
            b_bool: bool[V];
            in_bool: bool[V];
            a_bitset: bitset[V];
            b_bitset: bitset[V];
            in_bitset: bitset[V];
            a_i4: i4[V];
            b_i4: i4[V];
            in_i4: i4[V];
            a_i2: i2[V];
            b_i2: i2[V];
            in_i2: i2[V];
            a_i1: i1[V];
            b_i1: i1[V];
            in_i1: i1[V];
            a_u4: u4[V];
            b_u4: u4[V];
            in_u4: u4[V];
            a_u2: u2[V];
            b_u2: u2[V];
            in_u2: u2[V];
            a_u1: u1[V];
            b_u1: u1[V];
            in_u1: u1[V];

            ma_i8: i8[B, M, K];
            mb_i8: i8[B, K, N];
            in_ma_i8: i8[B, M, K];
            ma_i16: i16[B, M, K];
            mb_i16: i16[B, K, N];
            in_ma_i16: i16[B, M, K];
            ma_i32: i32[B, M, K];
            mb_i32: i32[B, K, N];
            in_ma_i32: i32[B, M, K];
            ma_i64: i64[B, M, K];
            mb_i64: i64[B, K, N];
            in_ma_i64: i64[B, M, K];
            ma_u8: u8[B, M, K];
            mb_u8: u8[B, K, N];
            in_ma_u8: u8[B, M, K];
            ma_u16: u16[B, M, K];
            mb_u16: u16[B, K, N];
            in_ma_u16: u16[B, M, K];
            ma_u32: u32[B, M, K];
            mb_u32: u32[B, K, N];
            in_ma_u32: u32[B, M, K];
            ma_u64: u64[B, M, K];
            mb_u64: u64[B, K, N];
            in_ma_u64: u64[B, M, K];
            ma_f16: f16[B, M, K];
            mb_f16: f16[B, K, N];
            in_ma_f16: f16[B, M, K];
            ma_f32: f32[B, M, K];
            mb_f32: f32[B, K, N];
            in_ma_f32: f32[B, M, K];
            ma_f64: f64[B, M, K];
            mb_f64: f64[B, K, N];
            in_ma_f64: f64[B, M, K];
            ma_bool: bool[B, M, K];
            mb_bool: bool[B, K, N];
            in_ma_bool: bool[B, M, K];
            ma_bitset: bitset[B, M, K];
            mb_bitset: bitset[B, K, N];
            in_ma_bitset: bitset[B, M, K];
        }

        volatile {
            add_i8: i8[V];
            mul_i8: i8[V];
            abs_i8: i8[V];
            relu_i8: i8[V];
            fill_i8: i8[V];
            add_i16: i16[V];
            mul_i16: i16[V];
            abs_i16: i16[V];
            relu_i16: i16[V];
            fill_i16: i16[V];
            add_i32: i32[V];
            mul_i32: i32[V];
            abs_i32: i32[V];
            relu_i32: i32[V];
            fill_i32: i32[V];
            add_i64: i64[V];
            mul_i64: i64[V];
            abs_i64: i64[V];
            relu_i64: i64[V];
            fill_i64: i64[V];
            add_u8: u8[V];
            mul_u8: u8[V];
            fill_u8: u8[V];
            add_u16: u16[V];
            mul_u16: u16[V];
            fill_u16: u16[V];
            add_u32: u32[V];
            mul_u32: u32[V];
            fill_u32: u32[V];
            add_u64: u64[V];
            mul_u64: u64[V];
            fill_u64: u64[V];
            add_f16: f16[V];
            mul_f16: f16[V];
            abs_f16: f16[V];
            relu_f16: f16[V];
            fill_f16: f16[V];
            finite_f16: bool;
            add_bf16: bf16[V];
            mul_bf16: bf16[V];
            abs_bf16: bf16[V];
            relu_bf16: bf16[V];
            fill_bf16: bf16[V];
            finite_bf16: bool;
            add_f8: f8[V];
            mul_f8: f8[V];
            abs_f8: f8[V];
            relu_f8: f8[V];
            fill_f8: f8[V];
            finite_f8: bool;
            add_f32: f32[V];
            mul_f32: f32[V];
            abs_f32: f32[V];
            relu_f32: f32[V];
            fill_f32: f32[V];
            finite_f32: bool;
            add_f64: f64[V];
            mul_f64: f64[V];
            abs_f64: f64[V];
            relu_f64: f64[V];
            fill_f64: f64[V];
            finite_f64: bool;
            add_bool: bool[V];
            mul_bool: bool[V];
            fill_bool: bool[V];
            add_bitset: bitset[V];
            mul_bitset: bitset[V];
            fill_bitset: bitset[V];
            add_i4: i4[V];
            mul_i4: i4[V];
            abs_i4: i4[V];
            relu_i4: i4[V];
            fill_i4: i4[V];
            add_i2: i2[V];
            mul_i2: i2[V];
            abs_i2: i2[V];
            fill_i2: i2[V];
            add_i1: i1[V];
            mul_i1: i1[V];
            abs_i1: i1[V];
            fill_i1: i1[V];
            add_u4: u4[V];
            mul_u4: u4[V];
            fill_u4: u4[V];
            add_u2: u2[V];
            mul_u2: u2[V];
            fill_u2: u2[V];
            add_u1: u1[V];
            mul_u1: u1[V];
            fill_u1: u1[V];

            mm_i8: i8[B, M, N];
            mm_i16: i16[B, M, N];
            mm_i32: i32[B, M, N];
            mm_i64: i64[B, M, N];
            mm_u8: u8[B, M, N];
            mm_u16: u16[B, M, N];
            mm_u32: u32[B, M, N];
            mm_u64: u64[B, M, N];
            mm_f16: f16[B, M, N];
            mm_f32: f32[B, M, N];
            mm_f64: f64[B, M, N];
            mm_bool: bool[B, M, N];
            mm_bitset: bitset[B, M, N];

            add_acc_i8: i16[V];
            add_acc_i16: i32[V];
            add_acc_i32: i64[V];
            add_acc_u8: u16[V];
            add_acc_u16: u32[V];
            add_acc_u32: u64[V];
            add_acc_i4: i8[V];
            add_acc_i2: i8[V];
            add_acc_i1: i8[V];
            add_acc_u4: u8[V];
            add_acc_u2: u8[V];
            add_acc_u1: u8[V];

            mul_acc_i8: i16[V];
            mul_acc_i16: i32[V];
            mul_acc_i32: i64[V];
            mul_acc_u8: u16[V];
            mul_acc_u16: u32[V];
            mul_acc_u32: u64[V];
            mul_acc_i4: i8[V];
            mul_acc_i2: i8[V];
            mul_acc_i1: i8[V];
            mul_acc_u4: u8[V];
            mul_acc_u2: u8[V];
            mul_acc_u1: u8[V];

            abs_acc_i8: i16[V];
            abs_acc_i16: i32[V];
            abs_acc_i32: i64[V];
            abs_acc_i4: i8[V];
            abs_acc_i2: i8[V];
            abs_acc_i1: i8[V];

            mm_acc_i8: i16[B, M, N];
            mm_acc_i16: i32[B, M, N];
            mm_acc_i32: i64[B, M, N];
            mm_acc_u8: u16[B, M, N];
            mm_acc_u16: u32[B, M, N];
            mm_acc_u32: u64[B, M, N];
        }

        block entry {
            op add(a_i8, b_i8) >> add_i8;
            op mul(a_i8, b_i8) >> mul_i8;
            op abs(a_i8) >> abs_i8;
            op relu(a_i8, alpha=1, clamp_max=2) >> relu_i8;
            op fill(a_i8, value=1) >> fill_i8;
            op add(a_i16, b_i16) >> add_i16;
            op mul(a_i16, b_i16) >> mul_i16;
            op abs(a_i16) >> abs_i16;
            op relu(a_i16, alpha=1, clamp_max=2) >> relu_i16;
            op fill(a_i16, value=1) >> fill_i16;
            op add(a_i32, b_i32) >> add_i32;
            op mul(a_i32, b_i32) >> mul_i32;
            op abs(a_i32) >> abs_i32;
            op relu(a_i32, alpha=1, clamp_max=2) >> relu_i32;
            op fill(a_i32, value=1) >> fill_i32;
            op add(a_i64, b_i64) >> add_i64;
            op mul(a_i64, b_i64) >> mul_i64;
            op abs(a_i64) >> abs_i64;
            op relu(a_i64, alpha=1, clamp_max=2) >> relu_i64;
            op fill(a_i64, value=1) >> fill_i64;
            op add(a_u8, b_u8) >> add_u8;
            op mul(a_u8, b_u8) >> mul_u8;
            op fill(a_u8, value=1) >> fill_u8;
            op add(a_u16, b_u16) >> add_u16;
            op mul(a_u16, b_u16) >> mul_u16;
            op fill(a_u16, value=1) >> fill_u16;
            op add(a_u32, b_u32) >> add_u32;
            op mul(a_u32, b_u32) >> mul_u32;
            op fill(a_u32, value=1) >> fill_u32;
            op add(a_u64, b_u64) >> add_u64;
            op mul(a_u64, b_u64) >> mul_u64;
            op fill(a_u64, value=1) >> fill_u64;
            op add(a_f16, b_f16) >> add_f16;
            op mul(a_f16, b_f16) >> mul_f16;
            op abs(a_f16) >> abs_f16;
            op relu(a_f16, alpha=0.1, clamp_max=2.0) >> relu_f16;
            op fill(a_f16, value=1.0) >> fill_f16;
            op is_finite(a_f16) >> finite_f16;
            op add(a_bf16, b_bf16) >> add_bf16;
            op mul(a_bf16, b_bf16) >> mul_bf16;
            op abs(a_bf16) >> abs_bf16;
            op relu(a_bf16, alpha=0.1, clamp_max=2.0) >> relu_bf16;
            op fill(a_bf16, value=1.0) >> fill_bf16;
            op is_finite(a_bf16) >> finite_bf16;
            op add(a_f8, b_f8) >> add_f8;
            op mul(a_f8, b_f8) >> mul_f8;
            op abs(a_f8) >> abs_f8;
            op relu(a_f8, alpha=0.1, clamp_max=2.0) >> relu_f8;
            op fill(a_f8, value=1.0) >> fill_f8;
            op is_finite(a_f8) >> finite_f8;
            op add(a_f32, b_f32) >> add_f32;
            op mul(a_f32, b_f32) >> mul_f32;
            op abs(a_f32) >> abs_f32;
            op relu(a_f32, alpha=0.1, clamp_max=2.0) >> relu_f32;
            op fill(a_f32, value=1.0) >> fill_f32;
            op is_finite(a_f32) >> finite_f32;
            op add(a_f64, b_f64) >> add_f64;
            op mul(a_f64, b_f64) >> mul_f64;
            op abs(a_f64) >> abs_f64;
            op relu(a_f64, alpha=0.1, clamp_max=2.0) >> relu_f64;
            op fill(a_f64, value=1.0) >> fill_f64;
            op is_finite(a_f64) >> finite_f64;
            op add(a_bool, b_bool) >> add_bool;
            op mul(a_bool, b_bool) >> mul_bool;
            op fill(a_bool, value=true) >> fill_bool;
            op add(a_bitset, b_bitset) >> add_bitset;
            op mul(a_bitset, b_bitset) >> mul_bitset;
            op fill(a_bitset, value=1) >> fill_bitset;
            op add(a_i4, b_i4) >> add_i4;
            op mul(a_i4, b_i4) >> mul_i4;
            op abs(a_i4) >> abs_i4;
            op relu(a_i4, alpha=1, clamp_max=2) >> relu_i4;
            op fill(a_i4, value=1) >> fill_i4;
            op add(a_i2, b_i2) >> add_i2;
            op mul(a_i2, b_i2) >> mul_i2;
            op abs(a_i2) >> abs_i2;
            op fill(a_i2, value=1) >> fill_i2;
            op add(a_i1, b_i1) >> add_i1;
            op mul(a_i1, b_i1) >> mul_i1;
            op abs(a_i1) >> abs_i1;
            op fill(a_i1, value=0) >> fill_i1;
            op add(a_u4, b_u4) >> add_u4;
            op mul(a_u4, b_u4) >> mul_u4;
            op fill(a_u4, value=1) >> fill_u4;
            op add(a_u2, b_u2) >> add_u2;
            op mul(a_u2, b_u2) >> mul_u2;
            op fill(a_u2, value=1) >> fill_u2;
            op add(a_u1, b_u1) >> add_u1;
            op mul(a_u1, b_u1) >> mul_u1;
            op fill(a_u1, value=1) >> fill_u1;

            op matmul(ma_i8, mb_i8) >> mm_i8;
            op matmul(ma_i16, mb_i16) >> mm_i16;
            op matmul(ma_i32, mb_i32) >> mm_i32;
            op matmul(ma_i64, mb_i64) >> mm_i64;
            op matmul(ma_u8, mb_u8) >> mm_u8;
            op matmul(ma_u16, mb_u16) >> mm_u16;
            op matmul(ma_u32, mb_u32) >> mm_u32;
            op matmul(ma_u64, mb_u64) >> mm_u64;
            op matmul(ma_f16, mb_f16) >> mm_f16;
            op matmul(ma_f32, mb_f32) >> mm_f32;
            op matmul(ma_f64, mb_f64) >> mm_f64;
            op matmul(ma_bool, mb_bool) >> mm_bool;
            op matmul(ma_bitset, mb_bitset) >> mm_bitset;

            op add(in_i8, b_i8) >> in_i8;
            op add(in_i16, b_i16) >> in_i16;
            op add(in_i32, b_i32) >> in_i32;
            op add(in_i64, b_i64) >> in_i64;
            op add(in_u8, b_u8) >> in_u8;
            op add(in_u16, b_u16) >> in_u16;
            op add(in_u32, b_u32) >> in_u32;
            op add(in_u64, b_u64) >> in_u64;
            op add(in_f16, b_f16) >> in_f16;
            op add(in_bf16, b_bf16) >> in_bf16;
            op add(in_f8, b_f8) >> in_f8;
            op add(in_f32, b_f32) >> in_f32;
            op add(in_f64, b_f64) >> in_f64;
            op add(in_bool, b_bool) >> in_bool;
            op add(in_bitset, b_bitset) >> in_bitset;
            op add(in_i4, b_i4) >> in_i4;
            op add(in_i2, b_i2) >> in_i2;
            op add(in_i1, b_i1) >> in_i1;
            op add(in_u4, b_u4) >> in_u4;
            op add(in_u2, b_u2) >> in_u2;
            op add(in_u1, b_u1) >> in_u1;

            op mul(in_i8, b_i8) >> in_i8;
            op mul(in_i16, b_i16) >> in_i16;
            op mul(in_i32, b_i32) >> in_i32;
            op mul(in_i64, b_i64) >> in_i64;
            op mul(in_u8, b_u8) >> in_u8;
            op mul(in_u16, b_u16) >> in_u16;
            op mul(in_u32, b_u32) >> in_u32;
            op mul(in_u64, b_u64) >> in_u64;
            op mul(in_f16, b_f16) >> in_f16;
            op mul(in_bf16, b_bf16) >> in_bf16;
            op mul(in_f8, b_f8) >> in_f8;
            op mul(in_f32, b_f32) >> in_f32;
            op mul(in_f64, b_f64) >> in_f64;
            op mul(in_bool, b_bool) >> in_bool;
            op mul(in_bitset, b_bitset) >> in_bitset;
            op mul(in_i4, b_i4) >> in_i4;
            op mul(in_i2, b_i2) >> in_i2;
            op mul(in_i1, b_i1) >> in_i1;
            op mul(in_u4, b_u4) >> in_u4;
            op mul(in_u2, b_u2) >> in_u2;
            op mul(in_u1, b_u1) >> in_u1;

            op abs(in_i8) >> in_i8;
            op abs(in_i16) >> in_i16;
            op abs(in_i32) >> in_i32;
            op abs(in_i64) >> in_i64;
            op abs(in_i4) >> in_i4;
            op abs(in_i2) >> in_i2;
            op abs(in_i1) >> in_i1;
            op abs(in_f16) >> in_f16;
            op abs(in_bf16) >> in_bf16;
            op abs(in_f8) >> in_f8;
            op abs(in_f32) >> in_f32;
            op abs(in_f64) >> in_f64;

            op relu(in_i8, alpha=1, clamp_max=2) >> in_i8;
            op relu(in_i16, alpha=1, clamp_max=2) >> in_i16;
            op relu(in_i32, alpha=1, clamp_max=2) >> in_i32;
            op relu(in_i64, alpha=1, clamp_max=2) >> in_i64;
            op relu(in_i4, alpha=1, clamp_max=2) >> in_i4;
            op relu(in_f16, alpha=0.1, clamp_max=2.0) >> in_f16;
            op relu(in_bf16, alpha=0.1, clamp_max=2.0) >> in_bf16;
            op relu(in_f8, alpha=0.1, clamp_max=2.0) >> in_f8;
            op relu(in_f32, alpha=0.1, clamp_max=2.0) >> in_f32;
            op relu(in_f64, alpha=0.1, clamp_max=2.0) >> in_f64;

            op fill(in_i8, value=1) >> in_i8;
            op fill(in_i16, value=1) >> in_i16;
            op fill(in_i32, value=1) >> in_i32;
            op fill(in_i64, value=1) >> in_i64;
            op fill(in_u8, value=1) >> in_u8;
            op fill(in_u16, value=1) >> in_u16;
            op fill(in_u32, value=1) >> in_u32;
            op fill(in_u64, value=1) >> in_u64;
            op fill(in_f16, value=1.0) >> in_f16;
            op fill(in_bf16, value=1.0) >> in_bf16;
            op fill(in_f8, value=1.0) >> in_f8;
            op fill(in_f32, value=1.0) >> in_f32;
            op fill(in_f64, value=1.0) >> in_f64;
            op fill(in_bool, value=true) >> in_bool;
            op fill(in_bitset, value=1) >> in_bitset;
            op fill(in_i4, value=1) >> in_i4;
            op fill(in_i2, value=1) >> in_i2;
            op fill(in_i1, value=0) >> in_i1;
            op fill(in_u4, value=1) >> in_u4;
            op fill(in_u2, value=1) >> in_u2;
            op fill(in_u1, value=1) >> in_u1;

            op matmul(in_ma_i8, mb_i8) >> in_ma_i8;
            op matmul(in_ma_i16, mb_i16) >> in_ma_i16;
            op matmul(in_ma_i32, mb_i32) >> in_ma_i32;
            op matmul(in_ma_i64, mb_i64) >> in_ma_i64;
            op matmul(in_ma_u8, mb_u8) >> in_ma_u8;
            op matmul(in_ma_u16, mb_u16) >> in_ma_u16;
            op matmul(in_ma_u32, mb_u32) >> in_ma_u32;
            op matmul(in_ma_u64, mb_u64) >> in_ma_u64;
            op matmul(in_ma_f16, mb_f16) >> in_ma_f16;
            op matmul(in_ma_f32, mb_f32) >> in_ma_f32;
            op matmul(in_ma_f64, mb_f64) >> in_ma_f64;
            op matmul(in_ma_bool, mb_bool) >> in_ma_bool;
            op matmul(in_ma_bitset, mb_bitset) >> in_ma_bitset;

            op add(a_i8, b_i8, acc=i16) >> add_acc_i8;
            op add(a_i16, b_i16, acc=i32) >> add_acc_i16;
            op add(a_i32, b_i32, acc=i64) >> add_acc_i32;
            op add(a_u8, b_u8, acc=u16) >> add_acc_u8;
            op add(a_u16, b_u16, acc=u32) >> add_acc_u16;
            op add(a_u32, b_u32, acc=u64) >> add_acc_u32;
            op add(a_i4, b_i4, acc=i8) >> add_acc_i4;
            op add(a_i2, b_i2, acc=i8) >> add_acc_i2;
            op add(a_i1, b_i1, acc=i8) >> add_acc_i1;
            op add(a_u4, b_u4, acc=u8) >> add_acc_u4;
            op add(a_u2, b_u2, acc=u8) >> add_acc_u2;
            op add(a_u1, b_u1, acc=u8) >> add_acc_u1;

            op mul(a_i8, b_i8, acc=i16) >> mul_acc_i8;
            op mul(a_i16, b_i16, acc=i32) >> mul_acc_i16;
            op mul(a_i32, b_i32, acc=i64) >> mul_acc_i32;
            op mul(a_u8, b_u8, acc=u16) >> mul_acc_u8;
            op mul(a_u16, b_u16, acc=u32) >> mul_acc_u16;
            op mul(a_u32, b_u32, acc=u64) >> mul_acc_u32;
            op mul(a_i4, b_i4, acc=i8) >> mul_acc_i4;
            op mul(a_i2, b_i2, acc=i8) >> mul_acc_i2;
            op mul(a_i1, b_i1, acc=i8) >> mul_acc_i1;
            op mul(a_u4, b_u4, acc=u8) >> mul_acc_u4;
            op mul(a_u2, b_u2, acc=u8) >> mul_acc_u2;
            op mul(a_u1, b_u1, acc=u8) >> mul_acc_u1;

            op abs(a_i8, acc=i16) >> abs_acc_i8;
            op abs(a_i16, acc=i32) >> abs_acc_i16;
            op abs(a_i32, acc=i64) >> abs_acc_i32;
            op abs(a_i4, acc=i8) >> abs_acc_i4;
            op abs(a_i2, acc=i8) >> abs_acc_i2;
            op abs(a_i1, acc=i8) >> abs_acc_i1;

            op matmul(ma_i8, mb_i8, acc=i16) >> mm_acc_i8;
            op matmul(ma_i16, mb_i16, acc=i32) >> mm_acc_i16;
            op matmul(ma_i32, mb_i32, acc=i64) >> mm_acc_i32;
            op matmul(ma_u8, mb_u8, acc=u16) >> mm_acc_u8;
            op matmul(ma_u16, mb_u16, acc=u32) >> mm_acc_u16;
            op matmul(ma_u32, mb_u32, acc=u64) >> mm_acc_u32;
            return;
        }
    };

    let v = model.size_of("V")?;
    let m = model.size_of("M")?;
    let k = model.size_of("K")?;
    let n = model.size_of("N")?;
    let b = model.size_of("B")?;
    let sim_cpu = Simulator::new(&model, &g, Device::Cpu)?;
    let mut exec_cpu = sim_cpu.make_executor()?;
    populate_exec(&mut exec_cpu, v, m, k, n, b)?;
    exec_cpu.step()?;
    let mut refs = HashMap::new();
    let mut float_refs = HashMap::new();
    for_each_tensor_output_collect!(
        collect_tensor_ref,
        collect_tensor_ref_float,
        &mut exec_cpu,
        &mut refs,
        &mut float_refs
    );
    for_each_scalar_output!(collect_scalar_ref, &mut exec_cpu, &mut refs);

    let sim = Simulator::new(&model, &g, device)?;
    let mut exec = sim.make_executor()?;
    populate_exec(&mut exec, v, m, k, n, b)?;
    exec.step()?;

    openinfer::log!("⚠️ == Pass but drift. ✅ == Pass with no drift. ❌ == Fail");

    for_each_tensor_output_validate!(
        validate_tensor,
        validate_tensor_float,
        &mut exec,
        &refs,
        &float_refs
    );
    for_each_scalar_output!(validate_scalar, &mut exec, &refs);

    openinfer::log!("ops_accumulate_inplace completed on {:?}", device);

    Ok(())
}
