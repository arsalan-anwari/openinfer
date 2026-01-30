use anyhow::{anyhow, Result};

use crate::ops::cpu::{
    abs, add, and, argmax_axis, argmin_axis, ceil, clamp, div, eq, fill, filter, floor, floor_div,
    fma, ge, gt, is_finite, is_inf, is_nan, is_neg, le, lt, matmul, max, max_axis, mean_axis, min,
    min_axis, mul, ne, neg, not, or, popcount, prod_axis, recip, relu, rem, round, shl, shr, sign,
    sub, sum_axis, trunc, xor,
};
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
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in sub::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in div::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in floor_div::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in rem::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in abs::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in min::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in max::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in clamp::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in floor::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in ceil::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in round::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in trunc::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in and::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in or::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in xor::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in not::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in shl::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in shr::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in popcount::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in eq::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in ne::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in lt::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in le::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in gt::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in ge::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in filter::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in is_nan::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in is_inf::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in is_neg::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in fma::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in sum_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in mean_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in prod_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in max_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in min_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in argmax_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in argmin_axis::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in neg::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in recip::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in sign::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in mul::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in matmul::registry::ENTRIES.iter() {
        map.insert(key.clone(), *kernel);
    }
    for (key, kernel) in relu::registry::ENTRIES.iter() {
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
