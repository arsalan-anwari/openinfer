use anyhow::{anyhow, Result};

use crate::ops::cpu::add;
use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::ops::registry::{KernelFn, OpKey};

pub fn lookup_kernel(key: OpKey) -> Result<KernelFn> {
    CPU_KERNELS
        .get(&key)
        .copied()
        .ok_or_else(|| anyhow!("unsupported cpu op {:?}", key))
}

static CPU_KERNELS: Lazy<HashMap<OpKey, KernelFn>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (key, kernel) in add::registry::ENTRIES {
        map.insert(*key, *kernel);
    }
    map
});
