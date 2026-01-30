use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const NEG_NORMAL_DTYPES: &[DType] = &[
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

pub const NEG_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const NEG_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: NEG_NORMAL_DTYPES,
    accumulate: NEG_ACC_INT_PAIRS,
};
