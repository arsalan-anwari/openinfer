use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const SIGN_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::I1,
    DType::I2,
    DType::I4,
];

pub const SIGN_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const SIGN_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: SIGN_NORMAL_DTYPES,
    accumulate: SIGN_ACC_INT_PAIRS,
};
