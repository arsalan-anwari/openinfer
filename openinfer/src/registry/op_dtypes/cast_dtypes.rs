use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

#[allow(unused)]
pub const CAST_NORMAL_DTYPES: &[DType] = &[
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::I1,
    DType::I2,
    DType::I4,
    DType::U1,
    DType::U2,
    DType::U4,
];

#[allow(unused)]
pub const CAST_OUTPUT_DTYPES: &[DType] = &[
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

#[allow(unused)]
pub const CAST_ACC_INT_PAIRS: &[(DType, DType)] = &[];

#[allow(unused)]
pub const CAST_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: CAST_NORMAL_DTYPES,
    accumulate: CAST_ACC_INT_PAIRS,
};
