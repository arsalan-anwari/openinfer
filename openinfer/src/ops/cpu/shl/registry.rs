use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{shl_inplace_dispatch, shl_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Shl, |mode| match mode {
        OpMode::Normal => Some(shl_normal_dispatch),
        OpMode::Inplace => Some(shl_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build shl cpu entries")
});
