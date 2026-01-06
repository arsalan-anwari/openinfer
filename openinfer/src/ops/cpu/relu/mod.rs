use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::timer::Timer;

pub mod registry;

pub fn relu_f32(attrs: &OpAttrs, a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let (negative_slope, clamp_max) = match attrs {
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => (attr_value_f32(negative_slope)?, attr_value_f32(clamp_max)?),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let mut y = if *x >= 0.0 { *x } else { *x * negative_slope };
        if y > clamp_max {
            y = clamp_max;
        }
        out.push(y);
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
