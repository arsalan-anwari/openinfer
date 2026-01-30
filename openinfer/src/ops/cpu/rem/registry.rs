use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{rem_inplace_dispatch, rem_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Rem, |mode| match mode {
        OpMode::Normal => Some(rem_normal_dispatch),
        OpMode::Inplace => Some(rem_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build rem cpu entries")
});
