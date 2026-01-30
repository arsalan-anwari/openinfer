use once_cell::sync::Lazy;

use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(crate::graph::OpKind::Mul, |mode| match mode {
        OpMode::Normal => Some(kernel::mul_normal_dispatch),
        OpMode::Inplace => Some(kernel::mul_inplace_dispatch),
        OpMode::Accumulate => Some(kernel::mul_accumulate_dispatch),
    })
    .expect("failed to build mul vulkan entries")
});
