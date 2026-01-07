use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;

use super::relu_f32;
use crate::ops::cpu::relu as cpu_relu;

pub fn lookup_kernel_cpu_avx2_relu(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::F32, [DType::F32], &OpAttrs::Relu { .. }) => {
            Some(KernelFn::Host(cpu_kernel(
                relu_f32 as fn(&OpAttrs, &[f32], usize) -> Result<Vec<f32>>,
            )))
        }
        (DType::F64, [DType::F64], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_f64 as fn(&OpAttrs, &[f64], usize) -> Result<Vec<f64>>,
        ))),
        (DType::I8, [DType::I8], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_i8 as fn(&OpAttrs, &[i8], usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_i16 as fn(&OpAttrs, &[i16], usize) -> Result<Vec<i16>>,
        ))),
        (DType::I32, [DType::I32], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_i32 as fn(&OpAttrs, &[i32], usize) -> Result<Vec<i32>>,
        ))),
        (DType::I64, [DType::I64], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_i64 as fn(&OpAttrs, &[i64], usize) -> Result<Vec<i64>>,
        ))),
        (DType::U8, [DType::U8], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_u8 as fn(&OpAttrs, &[u8], usize) -> Result<Vec<u8>>,
        ))),
        (DType::U16, [DType::U16], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_u16 as fn(&OpAttrs, &[u16], usize) -> Result<Vec<u16>>,
        ))),
        (DType::U32, [DType::U32], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_u32 as fn(&OpAttrs, &[u32], usize) -> Result<Vec<u32>>,
        ))),
        (DType::U64, [DType::U64], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_u64 as fn(&OpAttrs, &[u64], usize) -> Result<Vec<u64>>,
        ))),
        (DType::Bool, [DType::Bool], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            cpu_relu::relu_bool as fn(&OpAttrs, &[bool], usize) -> Result<Vec<bool>>,
        ))),
        _ => None,
    }
}
