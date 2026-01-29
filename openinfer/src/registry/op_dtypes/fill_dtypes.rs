use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

#[allow(dead_code)]
pub const FILL_ACC_PAIRS: &[(DType, DType)] = &[];

#[allow(dead_code)]
pub const FILL_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::BF16,
    DType::F16,
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
    DType::Bool,
    DType::Bitset,
    DType::I1,
    DType::I2,
    DType::I4,
    DType::U1,
    DType::U2,
    DType::U4,
];

pub const FILL_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: FILL_NORMAL_DTYPES,
    accumulate: FILL_ACC_PAIRS,
};
