use anyhow::{anyhow, Result};
use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::ops::registry::{KernelFn, OpKey};

use super::add;

pub fn lookup_kernel(key: OpKey) -> Result<KernelFn> {
    VULKAN_KERNELS
        .get(&key)
        .copied()
        .ok_or_else(|| anyhow!("unsupported vulkan op {:?}", key))
}

pub fn warm_kernels() {
    Lazy::force(&VULKAN_KERNELS);
}

static VULKAN_KERNELS: Lazy<HashMap<OpKey, KernelFn>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (key, kernel) in add::registry::ENTRIES.iter() {
        map.insert(*key, *kernel);
    }
    map
});
