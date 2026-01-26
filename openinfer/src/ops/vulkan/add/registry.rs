use crate::ops::registry::{KernelFn, OpMode};

use super::kernel;

pub fn kernel_for_mode(mode: OpMode) -> KernelFn {
    match mode {
        OpMode::Normal => kernel::dispatch_add_normal,
        OpMode::Inplace => kernel::dispatch_add_inplace,
        OpMode::Accumulate => kernel::dispatch_add_accumulate,
    }
}
