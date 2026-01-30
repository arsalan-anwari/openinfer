use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const FLOOR_DIV_NORMAL_DTYPES: &[DType] = &[
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::I4,
    DType::U4,
];

pub const FLOOR_DIV_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const FLOOR_DIV_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: FLOOR_DIV_NORMAL_DTYPES,
    accumulate: FLOOR_DIV_ACC_INT_PAIRS,
};
