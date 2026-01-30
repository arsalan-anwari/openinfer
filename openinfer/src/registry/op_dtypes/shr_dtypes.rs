use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const SHR_NORMAL_DTYPES: &[DType] = &[
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::I2,
    DType::I4,
    DType::U2,
    DType::U4,
];

pub const SHR_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const SHR_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: SHR_NORMAL_DTYPES,
    accumulate: SHR_ACC_INT_PAIRS,
};
