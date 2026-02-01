pub mod arithmetic;
pub mod bitwise;
pub mod casting;
pub mod comparison;
#[path = "filter/mod.rs"]
pub mod filtering;
pub mod mutation;
pub mod numerical;
pub mod reduction;
pub mod rounding;
pub mod statistics;

pub use arithmetic::{abs, add, div, floor_div, fma, mul, neg, recip, rem, sub};
pub use bitwise::{and, not, or, popcount, shl, shr, xor};
pub use casting::cast;
pub use comparison::{eq, ge, gt, le, lt, ne};
pub use filtering::{filter, is_finite, is_inf, is_nan, is_neg};
pub use mutation::fill;
pub use numerical::{matmul, relu};
pub use reduction::{
    argmax_axis, argmin_axis, max_axis, mean_axis, min_axis, prod_axis, sum_axis,
};
pub use rounding::{ceil, clamp, floor, round, trunc};
pub use statistics::{max, min, sign};
pub mod elementwise;
pub mod packed_ops;
pub mod accumulate;
pub mod compare;
pub mod reduce;
pub mod broadcast;
pub mod registry;
pub mod packed_cpu;
