use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{clamp_inplace_dispatch, clamp_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Clamp, |mode| match mode {
        OpMode::Normal => Some(clamp_normal_dispatch),
        OpMode::Inplace => Some(clamp_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build clamp cpu entries")
});
