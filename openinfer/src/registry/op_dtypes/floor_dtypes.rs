use crate::registry::OpDTypeSupport;
use crate::tensor::DType;

pub const FLOOR_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

pub const FLOOR_ACC_INT_PAIRS: &[(DType, DType)] = &[];

pub const FLOOR_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: FLOOR_NORMAL_DTYPES,
    accumulate: FLOOR_ACC_INT_PAIRS,
};
