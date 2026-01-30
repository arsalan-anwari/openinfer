use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const CEIL_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const CEIL_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const CEIL_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: CEIL_NORMAL_DTYPES,
    accumulate: CEIL_ACC_INT_PAIRS,
};
