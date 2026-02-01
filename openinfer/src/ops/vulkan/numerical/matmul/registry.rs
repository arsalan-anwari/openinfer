use once_cell::sync::Lazy;

use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(crate::graph::OpKind::Matmul, |mode| match mode {
        OpMode::Normal => Some(kernel::matmul_normal_dispatch),
        OpMode::Inplace => Some(kernel::matmul_inplace_dispatch),
        OpMode::Accumulate => Some(kernel::matmul_accumulate_dispatch),
    })
    .expect("failed to build matmul vulkan entries")
});
