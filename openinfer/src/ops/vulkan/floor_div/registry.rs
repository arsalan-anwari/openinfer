use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::FloorDiv, |mode| match mode {
        OpMode::Normal => Some(kernel::floor_div_normal_dispatch),
        OpMode::Inplace => Some(kernel::floor_div_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build floor_div vulkan entries")
});
