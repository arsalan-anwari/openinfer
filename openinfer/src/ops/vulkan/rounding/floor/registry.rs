use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Floor, |mode| match mode {
        OpMode::Normal => Some(kernel::floor_normal_dispatch),
        OpMode::Inplace => Some(kernel::floor_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build floor vulkan entries")
});
