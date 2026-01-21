use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;

use super::{add_f32, add_f64, add_i16, add_i8, add_u16, add_u8};

pub fn lookup_kernel_cpu_avx_add(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_i8 as fn(&[i8], &[i8] , usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_i16 as fn(&[i16], &[i16] , usize) -> Result<Vec<i16>>,
        ))),
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_f32 as fn(&[f32], &[f32] , usize) -> Result<Vec<f32>>,
        ))),
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_f64 as fn(&[f64], &[f64] , usize) -> Result<Vec<f64>>,
        ))),
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_u8 as fn(&[u8], &[u8] , usize) -> Result<Vec<u8>>,
        ))),
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_u16 as fn(&[u16], &[u16] , usize) -> Result<Vec<u16>>,
        ))),
        _ => None,
    }
}
