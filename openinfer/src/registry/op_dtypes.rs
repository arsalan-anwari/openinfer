use crate::tensor::DType;

pub mod abs_dtypes;
pub mod add_dtypes;
pub mod fill_dtypes;
pub mod is_finite_dtypes;
pub mod matmul_dtypes;
pub mod mul_dtypes;
pub mod relu_dtypes;

pub use abs_dtypes::{ABS_ACC_INT_PAIRS, ABS_DTYPE_SUPPORT, ABS_NORMAL_DTYPES};
pub use add_dtypes::{ADD_ACC_INT_PAIRS, ADD_DTYPE_SUPPORT, ADD_NORMAL_DTYPES};
pub use fill_dtypes::{FILL_ACC_PAIRS, FILL_DTYPE_SUPPORT, FILL_NORMAL_DTYPES};
pub use is_finite_dtypes::{IS_FINITE_ACC_PAIRS, IS_FINITE_DTYPE_SUPPORT, IS_FINITE_NORMAL_DTYPES};
pub use matmul_dtypes::{MATMUL_ACC_INT_PAIRS, MATMUL_DTYPE_SUPPORT, MATMUL_NORMAL_DTYPES};
pub use mul_dtypes::{MUL_ACC_INT_PAIRS, MUL_DTYPE_SUPPORT, MUL_NORMAL_DTYPES};
pub use relu_dtypes::{RELU_ACC_PAIRS, RELU_DTYPE_SUPPORT, RELU_NORMAL_DTYPES};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OpDTypeSupport {
    pub normal: &'static [DType],
    pub accumulate: &'static [(DType, DType)],
}

#[allow(dead_code)]
pub const ACC_INT_PAIRS: &[(DType, DType)] = &[
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
    (DType::U1, DType::U8),
    (DType::U1, DType::U16),
    (DType::U1, DType::U32),
    (DType::U1, DType::U64),
    (DType::U2, DType::U8),
    (DType::U2, DType::U16),
    (DType::U2, DType::U32),
    (DType::U2, DType::U64),
    (DType::U4, DType::U8),
    (DType::U4, DType::U16),
    (DType::U4, DType::U32),
    (DType::U4, DType::U64),
    (DType::U8, DType::U16),
    (DType::U8, DType::U32),
    (DType::U8, DType::U64),
    (DType::U16, DType::U32),
    (DType::U16, DType::U64),
    (DType::U32, DType::U64),
];
