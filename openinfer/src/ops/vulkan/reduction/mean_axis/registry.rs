use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::MeanAxis, |mode| match mode {
        OpMode::Normal => Some(kernel::mean_axis_normal_dispatch),
        OpMode::Inplace => None,
        OpMode::Accumulate => Some(kernel::mean_axis_accumulate_dispatch),
    })
    .expect("failed to build mean_axis vulkan entries")
});
