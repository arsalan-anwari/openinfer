use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const IS_NEG_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::I1,
    DType::I2,
    DType::I4,
];

pub const IS_NEG_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const IS_NEG_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: IS_NEG_NORMAL_DTYPES,
    accumulate: IS_NEG_ACC_INT_PAIRS,
};
