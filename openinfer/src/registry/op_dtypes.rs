use crate::tensor::DType;

pub mod abs_dtypes;
pub mod add_dtypes;
pub mod and_dtypes;
pub mod argmax_axis_dtypes;
pub mod argmin_axis_dtypes;
pub mod cast_dtypes;
pub mod ceil_dtypes;
pub mod clamp_dtypes;
pub mod div_dtypes;
pub mod eq_dtypes;
pub mod fill_dtypes;
pub mod filter_dtypes;
pub mod floor_div_dtypes;
pub mod floor_dtypes;
pub mod fma_dtypes;
pub mod ge_dtypes;
pub mod gt_dtypes;
pub mod is_finite_dtypes;
pub mod is_inf_dtypes;
pub mod is_nan_dtypes;
pub mod is_neg_dtypes;
pub mod le_dtypes;
pub mod lt_dtypes;
pub mod matmul_dtypes;
pub mod max_axis_dtypes;
pub mod max_dtypes;
pub mod mean_axis_dtypes;
pub mod min_axis_dtypes;
pub mod min_dtypes;
pub mod mul_dtypes;
pub mod ne_dtypes;
pub mod neg_dtypes;
pub mod not_dtypes;
pub mod or_dtypes;
pub mod popcount_dtypes;
pub mod prod_axis_dtypes;
pub mod recip_dtypes;
pub mod relu_dtypes;
pub mod rem_dtypes;
pub mod round_dtypes;
pub mod shl_dtypes;
pub mod shr_dtypes;
pub mod sign_dtypes;
pub mod sub_dtypes;
pub mod sum_axis_dtypes;
pub mod trunc_dtypes;
pub mod xor_dtypes;

pub use abs_dtypes::{ABS_ACC_INT_PAIRS, ABS_DTYPE_SUPPORT, ABS_NORMAL_DTYPES};
pub use add_dtypes::{ADD_ACC_INT_PAIRS, ADD_DTYPE_SUPPORT, ADD_NORMAL_DTYPES};
pub use and_dtypes::{AND_ACC_INT_PAIRS, AND_DTYPE_SUPPORT, AND_NORMAL_DTYPES};
pub use argmax_axis_dtypes::{
    ARGMAX_AXIS_ACC_INT_PAIRS, ARGMAX_AXIS_DTYPE_SUPPORT, ARGMAX_AXIS_NORMAL_DTYPES,
};
pub use argmin_axis_dtypes::{
    ARGMIN_AXIS_ACC_INT_PAIRS, ARGMIN_AXIS_DTYPE_SUPPORT, ARGMIN_AXIS_NORMAL_DTYPES,
};
pub use cast_dtypes::{CAST_ACC_INT_PAIRS, CAST_DTYPE_SUPPORT, CAST_NORMAL_DTYPES};
pub use ceil_dtypes::{CEIL_ACC_INT_PAIRS, CEIL_DTYPE_SUPPORT, CEIL_NORMAL_DTYPES};
pub use clamp_dtypes::{CLAMP_ACC_INT_PAIRS, CLAMP_DTYPE_SUPPORT, CLAMP_NORMAL_DTYPES};
pub use div_dtypes::{DIV_ACC_INT_PAIRS, DIV_DTYPE_SUPPORT, DIV_NORMAL_DTYPES};
pub use eq_dtypes::{EQ_ACC_INT_PAIRS, EQ_DTYPE_SUPPORT, EQ_NORMAL_DTYPES};
pub use fill_dtypes::{FILL_ACC_PAIRS, FILL_DTYPE_SUPPORT, FILL_NORMAL_DTYPES};
pub use filter_dtypes::{FILTER_ACC_INT_PAIRS, FILTER_DTYPE_SUPPORT, FILTER_NORMAL_DTYPES};
pub use floor_div_dtypes::{
    FLOOR_DIV_ACC_INT_PAIRS, FLOOR_DIV_DTYPE_SUPPORT, FLOOR_DIV_NORMAL_DTYPES,
};
pub use floor_dtypes::{FLOOR_ACC_INT_PAIRS, FLOOR_DTYPE_SUPPORT, FLOOR_NORMAL_DTYPES};
pub use fma_dtypes::{FMA_ACC_INT_PAIRS, FMA_DTYPE_SUPPORT, FMA_NORMAL_DTYPES};
pub use ge_dtypes::{GE_ACC_INT_PAIRS, GE_DTYPE_SUPPORT, GE_NORMAL_DTYPES};
pub use gt_dtypes::{GT_ACC_INT_PAIRS, GT_DTYPE_SUPPORT, GT_NORMAL_DTYPES};
pub use is_finite_dtypes::{IS_FINITE_ACC_PAIRS, IS_FINITE_DTYPE_SUPPORT, IS_FINITE_NORMAL_DTYPES};
pub use is_inf_dtypes::{IS_INF_ACC_INT_PAIRS, IS_INF_DTYPE_SUPPORT, IS_INF_NORMAL_DTYPES};
pub use is_nan_dtypes::{IS_NAN_ACC_INT_PAIRS, IS_NAN_DTYPE_SUPPORT, IS_NAN_NORMAL_DTYPES};
pub use is_neg_dtypes::{IS_NEG_ACC_INT_PAIRS, IS_NEG_DTYPE_SUPPORT, IS_NEG_NORMAL_DTYPES};
pub use le_dtypes::{LE_ACC_INT_PAIRS, LE_DTYPE_SUPPORT, LE_NORMAL_DTYPES};
pub use lt_dtypes::{LT_ACC_INT_PAIRS, LT_DTYPE_SUPPORT, LT_NORMAL_DTYPES};
pub use matmul_dtypes::{MATMUL_ACC_INT_PAIRS, MATMUL_DTYPE_SUPPORT, MATMUL_NORMAL_DTYPES};
pub use max_axis_dtypes::{MAX_AXIS_ACC_INT_PAIRS, MAX_AXIS_DTYPE_SUPPORT, MAX_AXIS_NORMAL_DTYPES};
pub use max_dtypes::{MAX_ACC_INT_PAIRS, MAX_DTYPE_SUPPORT, MAX_NORMAL_DTYPES};
pub use mean_axis_dtypes::{
    MEAN_AXIS_ACC_INT_PAIRS, MEAN_AXIS_DTYPE_SUPPORT, MEAN_AXIS_NORMAL_DTYPES,
};
pub use min_axis_dtypes::{MIN_AXIS_ACC_INT_PAIRS, MIN_AXIS_DTYPE_SUPPORT, MIN_AXIS_NORMAL_DTYPES};
pub use min_dtypes::{MIN_ACC_INT_PAIRS, MIN_DTYPE_SUPPORT, MIN_NORMAL_DTYPES};
pub use mul_dtypes::{MUL_ACC_INT_PAIRS, MUL_DTYPE_SUPPORT, MUL_NORMAL_DTYPES};
pub use ne_dtypes::{NE_ACC_INT_PAIRS, NE_DTYPE_SUPPORT, NE_NORMAL_DTYPES};
pub use neg_dtypes::{NEG_ACC_INT_PAIRS, NEG_DTYPE_SUPPORT, NEG_NORMAL_DTYPES};
pub use not_dtypes::{NOT_ACC_INT_PAIRS, NOT_DTYPE_SUPPORT, NOT_NORMAL_DTYPES};
pub use or_dtypes::{OR_ACC_INT_PAIRS, OR_DTYPE_SUPPORT, OR_NORMAL_DTYPES};
pub use popcount_dtypes::{
    POPCOUNT_ACC_INT_PAIRS, POPCOUNT_DTYPE_SUPPORT, POPCOUNT_NORMAL_DTYPES,
};
pub use prod_axis_dtypes::{
    PROD_AXIS_ACC_INT_PAIRS, PROD_AXIS_DTYPE_SUPPORT, PROD_AXIS_NORMAL_DTYPES,
};
pub use recip_dtypes::{RECIP_ACC_INT_PAIRS, RECIP_DTYPE_SUPPORT, RECIP_NORMAL_DTYPES};
pub use relu_dtypes::{RELU_ACC_PAIRS, RELU_DTYPE_SUPPORT, RELU_NORMAL_DTYPES};
pub use rem_dtypes::{REM_ACC_INT_PAIRS, REM_DTYPE_SUPPORT, REM_NORMAL_DTYPES};
pub use round_dtypes::{ROUND_ACC_INT_PAIRS, ROUND_DTYPE_SUPPORT, ROUND_NORMAL_DTYPES};
pub use shl_dtypes::{SHL_ACC_INT_PAIRS, SHL_DTYPE_SUPPORT, SHL_NORMAL_DTYPES};
pub use shr_dtypes::{SHR_ACC_INT_PAIRS, SHR_DTYPE_SUPPORT, SHR_NORMAL_DTYPES};
pub use sign_dtypes::{SIGN_ACC_INT_PAIRS, SIGN_DTYPE_SUPPORT, SIGN_NORMAL_DTYPES};
pub use sub_dtypes::{SUB_ACC_INT_PAIRS, SUB_DTYPE_SUPPORT, SUB_NORMAL_DTYPES};
pub use sum_axis_dtypes::{SUM_AXIS_ACC_INT_PAIRS, SUM_AXIS_DTYPE_SUPPORT, SUM_AXIS_NORMAL_DTYPES};
pub use trunc_dtypes::{TRUNC_ACC_INT_PAIRS, TRUNC_DTYPE_SUPPORT, TRUNC_NORMAL_DTYPES};
pub use xor_dtypes::{XOR_ACC_INT_PAIRS, XOR_DTYPE_SUPPORT, XOR_NORMAL_DTYPES};

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
