use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::ne_normal_dispatch;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Ne, |mode| match mode {
        OpMode::Normal => Some(ne_normal_dispatch),
        OpMode::Inplace => None,
        OpMode::Accumulate => None,
    })
    .expect("failed to build ne cpu entries")
});
