use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const XOR_NORMAL_DTYPES: &[DType] = &[
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

pub const XOR_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const XOR_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: XOR_NORMAL_DTYPES,
    accumulate: XOR_ACC_INT_PAIRS,
};
