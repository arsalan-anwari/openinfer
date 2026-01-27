use once_cell::sync::Lazy;

use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(crate::graph::OpKind::Add, |mode| {
        Some(kernel_for_mode(mode))
    })
    .expect("failed to build add vulkan entries")
});

pub fn kernel_for_mode(mode: OpMode) -> KernelFn {
    match mode {
        OpMode::Normal => kernel::dispatch_add_normal,
        OpMode::Inplace => kernel::dispatch_add_inplace,
        OpMode::Accumulate => kernel::dispatch_add_accumulate,
    }
}

