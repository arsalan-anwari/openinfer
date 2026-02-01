use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{fill_inplace_dispatch, fill_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Fill, |mode| match mode {
        OpMode::Normal => Some(fill_normal_dispatch),
        OpMode::Inplace => Some(fill_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build fill cpu entries")
});
