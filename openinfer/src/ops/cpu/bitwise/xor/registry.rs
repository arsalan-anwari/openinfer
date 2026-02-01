use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{xor_inplace_dispatch, xor_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Xor, |mode| match mode {
        OpMode::Normal => Some(xor_normal_dispatch),
        OpMode::Inplace => Some(xor_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build xor cpu entries")
});
