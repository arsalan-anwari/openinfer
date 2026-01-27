use anyhow::{anyhow, Result};

use crate::ops::cpu::add;
use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::ops::registry::{KernelFn, OpKey};
use crate::tensor::TensorValue;

pub fn lookup_kernel(key: OpKey) -> Result<KernelFn> {
    CPU_KERNELS
        .get(&key)
        .copied()
        .ok_or_else(|| anyhow!("unsupported cpu op {:?}", key))
}

pub fn warm_kernels() {
    Lazy::force(&CPU_KERNELS);
}

#[allow(unused)]
pub fn expect_output(output: Option<&mut TensorValue>) -> Result<&mut TensorValue> {
    output.ok_or_else(|| anyhow!("missing output tensor"))
}

static CPU_KERNELS: Lazy<HashMap<OpKey, KernelFn>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (key, kernel) in add::registry::ENTRIES.iter() {
        map.insert(*key, *kernel);
    }
    map
});
