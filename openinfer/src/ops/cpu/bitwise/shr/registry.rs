use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{shr_inplace_dispatch, shr_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Shr, |mode| match mode {
        OpMode::Normal => Some(shr_normal_dispatch),
        OpMode::Inplace => Some(shr_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build shr cpu entries")
});
