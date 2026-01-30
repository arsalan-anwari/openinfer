use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const REM_NORMAL_DTYPES: &[DType] = &[
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

pub const REM_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const REM_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: REM_NORMAL_DTYPES,
    accumulate: REM_ACC_INT_PAIRS,
};
