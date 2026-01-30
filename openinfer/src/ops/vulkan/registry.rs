use anyhow::{anyhow, Result};
use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::ops::registry::{KernelFn, OpKey};

use super::{abs, add, fill, is_finite, matmul, mul, relu};

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
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in mul::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in abs::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in relu::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in matmul::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in is_finite::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in fill::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    map
});
