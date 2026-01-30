use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const ROUND_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const ROUND_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const ROUND_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: ROUND_NORMAL_DTYPES,
    accumulate: ROUND_ACC_INT_PAIRS,
};
