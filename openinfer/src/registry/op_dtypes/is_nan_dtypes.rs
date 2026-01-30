use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const IS_NAN_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const IS_NAN_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const IS_NAN_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: IS_NAN_NORMAL_DTYPES,
    accumulate: IS_NAN_ACC_INT_PAIRS,
};
