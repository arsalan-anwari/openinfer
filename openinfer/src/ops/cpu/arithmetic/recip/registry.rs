use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{recip_inplace_dispatch, recip_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Recip, |mode| match mode {
        OpMode::Normal => Some(recip_normal_dispatch),
        OpMode::Inplace => Some(recip_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build recip cpu entries")
});
