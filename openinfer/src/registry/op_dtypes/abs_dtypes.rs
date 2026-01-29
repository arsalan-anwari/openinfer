use crate::tensor::DType;
use crate::registry::OpDTypeSupport;

// [f8, bf16, f16, f32, f64] [i1, i2, i4, i8, i16, i32, i64]

#[allow(dead_code)]
pub const ABS_ACC_INT_PAIRS: &[(DType, DType)] = &[
    (DType::I1, DType::I8),
    (DType::I1, DType::I16),
    (DType::I1, DType::I32),
    (DType::I1, DType::I64),
    (DType::I2, DType::I8),
    (DType::I2, DType::I16),
    (DType::I2, DType::I32),
    (DType::I2, DType::I64),
    (DType::I4, DType::I8),
    (DType::I4, DType::I16),
    (DType::I4, DType::I32),
    (DType::I4, DType::I64),
    (DType::I8, DType::I16),
    (DType::I8, DType::I32),
    (DType::I8, DType::I64),
    (DType::I16, DType::I32),
    (DType::I16, DType::I64),
    (DType::I32, DType::I64),
];

#[allow(dead_code)]
pub const ABS_NORMAL_DTYPES: &[DType] = &[
    DType::F8,
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::I1,
    DType::I2,
    DType::I4,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64
];



pub const ABS_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: ABS_NORMAL_DTYPES,
    accumulate: ABS_ACC_INT_PAIRS,
};
