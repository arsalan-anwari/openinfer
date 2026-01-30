use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const RECIP_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const RECIP_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const RECIP_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: RECIP_NORMAL_DTYPES,
    accumulate: RECIP_ACC_INT_PAIRS,
};
