use once_cell::sync::Lazy;

use crate::ops::registry::{KernelFn, OpKey, OpMode};
use crate::registry::ACC_INT_PAIRS;

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    let mut entries = Vec::new();
    for (key, _) in crate::ops::cpu::add::registry::ENTRIES {
        if key.mode != OpMode::Accumulate {
            entries.push((*key, kernel_for_mode(key.mode)));
        }
    }
    for (in_dtype, out_dtype) in ACC_INT_PAIRS.iter().copied() {
        let key = OpKey {
            kind: crate::graph::OpKind::Add,
            mode: OpMode::Accumulate,
            broadcast: false,
            in0: in_dtype,
            in1: Some(in_dtype),
            out0: out_dtype,
        };
        entries.push((key, kernel_for_mode(OpMode::Accumulate)));
        let broadcast_key = OpKey { broadcast: true, ..key };
        entries.push((broadcast_key, kernel_for_mode(OpMode::Accumulate)));
    }
    entries
});

pub fn kernel_for_mode(mode: OpMode) -> KernelFn {
    match mode {
        OpMode::Normal => kernel::dispatch_add_normal,
        OpMode::Inplace => kernel::dispatch_add_inplace,
        OpMode::Accumulate => kernel::dispatch_add_accumulate,
    }
}

