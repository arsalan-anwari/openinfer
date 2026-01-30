use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const FMA_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const FMA_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const FMA_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: FMA_NORMAL_DTYPES,
    accumulate: FMA_ACC_INT_PAIRS,
};
