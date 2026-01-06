use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;

use super::{
    abs_bitset, abs_bool, abs_f16, abs_f32, abs_f64, abs_i16, abs_i32, abs_i64, abs_i8, abs_u16,
    abs_u32, abs_u64, abs_u8,
};

pub fn lookup_kernel_cpu_avx_abs(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i8 as fn(&[i8] , usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i16 as fn(&[i16] , usize) -> Result<Vec<i16>>,
        ))),
        (DType::F32, [DType::F32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_f32 as fn(&[f32] , usize) -> Result<Vec<f32>>,
        ))),
        (DType::F64, [DType::F64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_f64 as fn(&[f64] , usize) -> Result<Vec<f64>>,
        ))),
        (DType::U8, [DType::U8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_u8 as fn(&[u8] , usize) -> Result<Vec<u8>>,
        ))),
        (DType::U16, [DType::U16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_u16 as fn(&[u16] , usize) -> Result<Vec<u16>>,
        ))),
        (DType::I32, [DType::I32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i32 as fn(&[i32] , usize) -> Result<Vec<i32>>,
        ))),
        (DType::I64, [DType::I64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i64 as fn(&[i64] , usize) -> Result<Vec<i64>>,
        ))),
        (DType::U32, [DType::U32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_u32 as fn(&[u32] , usize) -> Result<Vec<u32>>,
        ))),
        (DType::U64, [DType::U64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_u64 as fn(&[u64] , usize) -> Result<Vec<u64>>,
        ))),
        (DType::Bool, [DType::Bool], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_bool as fn(&[bool] , usize) -> Result<Vec<bool>>,
        ))),
        (DType::Bitset, [DType::Bitset], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_bitset as fn(&[crate::tensor::Bitset] , usize) -> Result<Vec<crate::tensor::Bitset>>,
        ))),
        (DType::F16, [DType::F16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_f16 as fn(&[crate::tensor::F16] , usize) -> Result<Vec<crate::tensor::F16>>,
        ))),
        _ => None,
    }
}
