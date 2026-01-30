use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const TRUNC_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const TRUNC_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const TRUNC_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: TRUNC_NORMAL_DTYPES,
    accumulate: TRUNC_ACC_INT_PAIRS,
};
