use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::SumAxis, |mode| match mode {
        OpMode::Normal => Some(kernel::sum_axis_normal_dispatch),
        OpMode::Inplace => None,
        OpMode::Accumulate => Some(kernel::sum_axis_accumulate_dispatch),
    })
    .expect("failed to build sum_axis vulkan entries")
});
