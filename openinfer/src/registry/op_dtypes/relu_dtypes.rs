use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

#[allow(dead_code)]
pub const RELU_ACC_PAIRS: &[(DType, DType)] = &[];

#[allow(dead_code)]
pub const RELU_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::I4,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
];

pub const RELU_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: RELU_NORMAL_DTYPES,
    accumulate: RELU_ACC_PAIRS,
};
