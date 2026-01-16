use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;

use super::{
    fill_bitset, fill_bool, fill_f16, fill_f32, fill_f64, fill_i16, fill_i32, fill_i64, fill_i8,
    fill_u16, fill_u32, fill_u64, fill_u8,
};

pub fn lookup_kernel_cpu_fill(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i8 as fn(&OpAttrs, &[i8], usize) -> Result<Vec<i8>>),
        )),
        (DType::I16, [DType::I16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i16 as fn(&OpAttrs, &[i16], usize) -> Result<Vec<i16>>),
        )),
        (DType::I32, [DType::I32], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i32 as fn(&OpAttrs, &[i32], usize) -> Result<Vec<i32>>),
        )),
        (DType::I64, [DType::I64], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i64 as fn(&OpAttrs, &[i64], usize) -> Result<Vec<i64>>),
        )),
        (DType::U8, [DType::U8], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u8 as fn(&OpAttrs, &[u8], usize) -> Result<Vec<u8>>),
        )),
        (DType::U16, [DType::U16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u16 as fn(&OpAttrs, &[u16], usize) -> Result<Vec<u16>>),
        )),
        (DType::U32, [DType::U32], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u32 as fn(&OpAttrs, &[u32], usize) -> Result<Vec<u32>>),
        )),
        (DType::U64, [DType::U64], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u64 as fn(&OpAttrs, &[u64], usize) -> Result<Vec<u64>>),
        )),
        (DType::F16, [DType::F16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f16 as fn(&OpAttrs, &[crate::tensor::F16], usize) -> Result<Vec<crate::tensor::F16>>),
        )),
        (DType::F32, [DType::F32], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f32 as fn(&OpAttrs, &[f32], usize) -> Result<Vec<f32>>),
        )),
        (DType::F64, [DType::F64], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f64 as fn(&OpAttrs, &[f64], usize) -> Result<Vec<f64>>),
        )),
        (DType::Bool, [DType::Bool], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_bool as fn(&OpAttrs, &[bool], usize) -> Result<Vec<bool>>),
        )),
        (DType::Bitset, [DType::Bitset], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(
                fill_bitset
                    as fn(
                        &OpAttrs,
                        &[crate::tensor::Bitset],
                        usize,
                    ) -> Result<Vec<crate::tensor::Bitset>>,
            ),
        )),
        _ => None,
    }
}
