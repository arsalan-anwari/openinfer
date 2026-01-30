use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const DIV_NORMAL_DTYPES: &[DType] = &[
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
    DType::I4,
    DType::U4,
];

pub const DIV_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const DIV_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: DIV_NORMAL_DTYPES,
    accumulate: DIV_ACC_INT_PAIRS,
};
