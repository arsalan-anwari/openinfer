use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{matmul_accumulate_dispatch, matmul_inplace_dispatch, matmul_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Matmul, |mode| match mode {
        OpMode::Normal => Some(matmul_normal_dispatch),
        OpMode::Inplace => Some(matmul_inplace_dispatch),
        OpMode::Accumulate => Some(matmul_accumulate_dispatch),
    })
    .expect("failed to build matmul cpu entries")
});
