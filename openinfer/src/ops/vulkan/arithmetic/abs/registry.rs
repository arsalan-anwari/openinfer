use once_cell::sync::Lazy;

use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(crate::graph::OpKind::Abs, |mode| match mode {
        OpMode::Normal => Some(kernel::abs_normal_dispatch),
        OpMode::Inplace => Some(kernel::abs_inplace_dispatch),
        OpMode::Accumulate => Some(kernel::abs_accumulate_dispatch),
    })
    .expect("failed to build abs vulkan entries")
});
