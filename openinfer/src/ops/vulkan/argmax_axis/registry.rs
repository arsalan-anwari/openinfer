use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::ArgmaxAxis, |mode| match mode {
        OpMode::Normal => Some(kernel::argmax_axis_normal_dispatch),
        OpMode::Inplace | OpMode::Accumulate => None,
    })
    .expect("failed to build argmax_axis vulkan entries")
});
