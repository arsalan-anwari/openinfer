use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    _mm256_blendv_ps, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_storeu_ps, _CMP_GE_OQ, _CMP_GT_OQ,
};

use crate::graph::{AttrValue, OpAttrs};
use crate::timer::Timer;

pub fn relu_f32(attrs: &OpAttrs, a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let (negative_slope, clamp_max) = match attrs {
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => (attr_value_f32(negative_slope)?, attr_value_f32(clamp_max)?),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let len = a.len();
    let mut out = vec![0.0f32; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm256_set1_ps(0.0);
        let slope = _mm256_set1_ps(negative_slope);
        let clamp = _mm256_set1_ps(clamp_max);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let mask = _mm256_cmp_ps(va, zero, _CMP_GE_OQ);
            let vneg = _mm256_mul_ps(va, slope);
            let vsel = _mm256_blendv_ps(vneg, va, mask);
            let clamp_mask = _mm256_cmp_ps(vsel, clamp, _CMP_GT_OQ);
            let vclamp = _mm256_blendv_ps(vsel, clamp, clamp_mask);
            _mm256_storeu_ps(out_ptr.add(i), vclamp);
            i += 8;
        }
        while i < len {
            let x = a[i];
            let mut y = if x >= 0.0 { x } else { x * negative_slope };
            if y > clamp_max {
                y = clamp_max;
            }
            *out_ptr.add(i) = y;
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Literal(val) => Ok(*val),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
