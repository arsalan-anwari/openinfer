use crate::tensor::{compute_strides, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum BroadcastVariant {
    Standard,
    Inplace,
    Accumulate,
}

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == compute_strides(shape)
}

pub fn needs_broadcast<T, O>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &Tensor<O>,
    _variant: Option<BroadcastVariant>,
) -> bool {
    a.shape() != b.shape()
        || out.shape() != a.shape()
        || !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
}

#[macro_export]
macro_rules! add_kernel {
    (Standard, $op:expr, $dtype:ty, $variant:ident, $func:ident, $broadcast_func:ident) => {
        Some(KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.len() < 2 {
                return Err(anyhow!("{} op expects 2 inputs", $op));
            }
            let a = <$dtype as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let b = <$dtype as TensorElement>::from_value(&inputs[1])
                .ok_or_else(|| anyhow!("{} input 1 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} op requires preallocated output", $op)),
            };
            if crate::ops::cpu::registry_helpers::needs_broadcast(
                &a,
                &b,
                &out,
                Some(crate::ops::cpu::registry_helpers::BroadcastVariant::Standard),
            ) {
                $broadcast_func(&a, &b, &mut out, thread_id)?;
            } else {
                $func(&a, &b, &mut out, thread_id)?;
            }
            Ok(None)
        })))
    };
    (BinaryNoBroadcast, $op:expr, $dtype:ty, $variant:ident, $func:ident) => {
        Some(KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.len() < 2 {
                return Err(anyhow!("{} op expects 2 inputs", $op));
            }
            let a = <$dtype as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let b = <$dtype as TensorElement>::from_value(&inputs[1])
                .ok_or_else(|| anyhow!("{} input 1 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} op requires preallocated output", $op)),
            };
            $func(&a, &b, &mut out, thread_id)?;
            Ok(None)
        })))
    };
    (Inplace, $op:expr, $variant:ident, $func:ident, $broadcast_func:ident) => {
        Some(InplaceKernelFn::Host(Box::new(|_attrs, output, inputs, thread_id| {
            let other = inputs
                .get(0)
                .ok_or_else(|| anyhow!("inplace {} expects at least 1 input", $op))?;
            match (output, other) {
                (TensorValue::$variant(ref mut out), TensorValue::$variant(b)) => {
                    if crate::ops::cpu::registry_helpers::needs_broadcast(
                        out,
                        b,
                        out,
                        Some(crate::ops::cpu::registry_helpers::BroadcastVariant::Inplace),
                    ) {
                        $broadcast_func(out, b, thread_id)
                    } else {
                        $func(out, b, thread_id)
                    }
                }
                _ => Err(anyhow!("inplace {} dtype mismatch", $op)),
            }
        })))
    };
    (InplaceBinaryNoBroadcast, $op:expr, $variant:ident, $func:ident) => {
        Some(InplaceKernelFn::Host(Box::new(|_attrs, output, inputs, thread_id| {
            let other = inputs
                .get(0)
                .ok_or_else(|| anyhow!("inplace {} expects at least 1 input", $op))?;
            match (output, other) {
                (TensorValue::$variant(ref mut out), TensorValue::$variant(b)) => {
                    $func(out, b, thread_id)
                }
                _ => Err(anyhow!("inplace {} dtype mismatch", $op)),
            }
        })))
    };
    (Accumulate, $op:expr, $in:ty, $out_variant:ident, $func:ident, $broadcast_func:ident) => {
        Some(KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.len() < 2 {
                return Err(anyhow!("{} op expects 2 inputs", $op));
            }
            let a = <$in as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let b = <$in as TensorElement>::from_value(&inputs[1])
                .ok_or_else(|| anyhow!("{} input 1 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$out_variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} accumulate requires preallocated output", $op)),
            };
            if crate::ops::cpu::registry_helpers::needs_broadcast(
                &a,
                &b,
                &out,
                Some(crate::ops::cpu::registry_helpers::BroadcastVariant::Accumulate),
            ) {
                $broadcast_func(&a, &b, &mut out, thread_id)?;
            } else {
                $func(&a, &b, &mut out, thread_id)?;
            }
            Ok(None)
        })))
    };
    (AccumulateBinaryNoBroadcast, $op:expr, $in:ty, $out_variant:ident, $func:ident) => {
        Some(KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.len() < 2 {
                return Err(anyhow!("{} op expects 2 inputs", $op));
            }
            let a = <$in as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let b = <$in as TensorElement>::from_value(&inputs[1])
                .ok_or_else(|| anyhow!("{} input 1 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$out_variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} accumulate requires preallocated output", $op)),
            };
            $func(&a, &b, &mut out, thread_id)?;
            Ok(None)
        })))
    };
    (Unary, $op:expr, $dtype:ty, $variant:ident, $func:ident) => {
        Some(KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.is_empty() {
                return Err(anyhow!("{} op expects 1 input", $op));
            }
            let a = <$dtype as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} op requires preallocated output", $op)),
            };
            $func(&a, &mut out, thread_id)?;
            Ok(None)
        })))
    };
    (UnaryAttrs, $op:expr, $dtype:ty, $variant:ident, $func:ident) => {
        Some(KernelFn::Host(Box::new(|attrs, inputs, output, thread_id| {
            if inputs.is_empty() {
                return Err(anyhow!("{} op expects 1 input", $op));
            }
            let a = <$dtype as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} op requires preallocated output", $op)),
            };
            $func(attrs, &a, &mut out, thread_id)?;
            Ok(None)
        })))
    };
    (InplaceUnary, $op:expr, $variant:ident, $func:ident) => {
        Some(InplaceKernelFn::Host(Box::new(|_attrs, output, _inputs, thread_id| {
            match output {
                TensorValue::$variant(ref mut out) => $func(out, thread_id),
                _ => Err(anyhow!("inplace {} dtype mismatch", $op)),
            }
        })))
    };
    (InplaceUnaryAttrs, $op:expr, $variant:ident, $func:ident) => {
        Some(InplaceKernelFn::Host(Box::new(|attrs, output, _inputs, thread_id| {
            match output {
                TensorValue::$variant(ref mut out) => $func(attrs, out, thread_id),
                _ => Err(anyhow!("inplace {} dtype mismatch", $op)),
            }
        })))
    };
    (AccumulateUnary, $op:expr, $in:ty, $out_variant:ident, $func:ident) => {
        Some(KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.is_empty() {
                return Err(anyhow!("{} op expects 1 input", $op));
            }
            let a = <$in as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow!("{} input 0 dtype mismatch", $op))?;
            let mut out = match output {
                Some(TensorValue::$out_variant(tensor)) => tensor,
                Some(_) => return Err(anyhow!("{} output dtype mismatch", $op)),
                None => return Err(anyhow!("{} accumulate requires preallocated output", $op)),
            };
            $func(&a, &mut out, thread_id)?;
            Ok(None)
        })))
    };
}
