use once_cell::sync::Lazy;

use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(crate::graph::OpKind::Relu, |mode| match mode {
        OpMode::Normal => Some(kernel::relu_normal_dispatch),
        OpMode::Inplace => Some(kernel::relu_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build relu vulkan entries")
});
