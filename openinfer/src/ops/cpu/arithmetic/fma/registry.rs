use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{fma_inplace_dispatch, fma_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Fma, |mode| match mode {
        OpMode::Normal => Some(fma_normal_dispatch),
        OpMode::Inplace => Some(fma_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build fma cpu entries")
});
