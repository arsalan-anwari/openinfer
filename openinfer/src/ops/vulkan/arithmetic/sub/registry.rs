use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Sub, |mode| match mode {
        OpMode::Normal => Some(kernel::sub_normal_dispatch),
        OpMode::Inplace => Some(kernel::sub_inplace_dispatch),
        OpMode::Accumulate => Some(kernel::sub_accumulate_dispatch),
    })
    .expect("failed to build sub vulkan entries")
});
