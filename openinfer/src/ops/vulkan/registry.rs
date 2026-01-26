use anyhow::{anyhow, Result};
use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::ops::registry::{KernelFn, OpKey, OpMode};
use crate::registry::{ACC_INT_PAIRS, PACKED_ACC_PAIRS};

use super::add;

pub fn lookup_kernel(key: OpKey) -> Result<KernelFn> {
    VULKAN_KERNELS
        .get(&key)
        .copied()
        .ok_or_else(|| anyhow!("unsupported vulkan op {:?}", key))
}

static VULKAN_KERNELS: Lazy<HashMap<OpKey, KernelFn>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (key, _) in crate::ops::cpu::add::registry::ENTRIES {
        if key.mode != OpMode::Accumulate {
            let kernel = add::registry::kernel_for_mode(key.mode);
            map.insert(*key, kernel);
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
        map.insert(key, add::registry::kernel_for_mode(OpMode::Accumulate));
        let broadcast_key = OpKey {
            broadcast: true,
            ..key
        };
        map.insert(broadcast_key, add::registry::kernel_for_mode(OpMode::Accumulate));
    }
    for (in_dtype, out_dtype) in PACKED_ACC_PAIRS.iter().copied() {
        let key = OpKey {
            kind: crate::graph::OpKind::Add,
            mode: OpMode::Accumulate,
            broadcast: false,
            in0: in_dtype,
            in1: Some(in_dtype),
            out0: out_dtype,
        };
        map.insert(key, add::registry::kernel_for_mode(OpMode::Accumulate));
        let broadcast_key = OpKey {
            broadcast: true,
            ..key
        };
        map.insert(broadcast_key, add::registry::kernel_for_mode(OpMode::Accumulate));
    }
    map
});
