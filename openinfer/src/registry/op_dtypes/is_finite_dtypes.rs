use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

#[allow(dead_code)]
pub const IS_FINITE_ACC_PAIRS: &[(DType, DType)] = &[];

#[allow(dead_code)]
pub const IS_FINITE_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
];

pub const IS_FINITE_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: IS_FINITE_NORMAL_DTYPES,
    accumulate: IS_FINITE_ACC_PAIRS,
};
