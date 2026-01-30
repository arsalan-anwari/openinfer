use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const OR_NORMAL_DTYPES: &[DType] = &[
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
    DType::U1,
    DType::U2,
    DType::U4,
    DType::Bool,
];

pub const OR_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const OR_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: OR_NORMAL_DTYPES,
    accumulate: OR_ACC_INT_PAIRS,
};
