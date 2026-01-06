use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::registry::HostKernel;
use crate::tensor::{TensorElement, TensorValue};

#[cfg(feature = "vulkan")]
use crate::backend::VulkanBuffer;
#[cfg(feature = "vulkan")]
use crate::ops::registry::VulkanKernel;

pub trait CpuKernelAdapter {
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue], thread_id: usize) -> Result<TensorValue>;
}

#[cfg(feature = "vulkan")]
pub trait DeviceKernelAdapter {
    fn call(
        &self,
        attrs: &OpAttrs,
        inputs: &[&VulkanBuffer],
        thread_id: usize,
    ) -> Result<VulkanBuffer>;
}

pub fn cpu_kernel<K>(func: K) -> HostKernel
where
    K: CpuKernelAdapter + Send + Sync + 'static,
{
    Box::new(move |attrs, inputs, thread_id| func.call(attrs, inputs, thread_id))
}

#[cfg(feature = "vulkan")]
pub fn device_kernel<K>(func: K) -> VulkanKernel
where
    K: DeviceKernelAdapter + Send + Sync + 'static,
{
    Box::new(move |attrs, inputs, thread_id| func.call(attrs, inputs, thread_id))
}

impl<A, R> CpuKernelAdapter for for<'a> fn(&'a [A], usize) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[TensorValue], thread_id: usize) -> Result<TensorValue> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("cpu kernel expects at least 1 input"))?;
        let a = A::from_value(a)
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        Ok((self)(&a.data, thread_id)?.into())
    }
}

impl<A, R> CpuKernelAdapter for for<'a, 'b> fn(&'a OpAttrs, &'b [A], usize) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue], thread_id: usize) -> Result<TensorValue> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("cpu kernel expects at least 1 input"))?;
        let a = A::from_value(a)
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        Ok((self)(attrs, &a.data, thread_id)?.into())
    }
}

macro_rules! impl_cpu_kernel_adapter_noattrs {
    ($count:literal, $(($var:ident, $T:ident, $lt:tt, $idx:tt)),+; $($all_lt:tt),+) => {
        impl<$($T,)+ R> CpuKernelAdapter for for<$($all_lt,)+> fn($( &$lt [$T], )+ usize) -> Result<R>
        where
            $($T: TensorElement + Clone + 'static,)+
            R: Into<TensorValue>,
        {
            fn call(&self, _attrs: &OpAttrs, inputs: &[TensorValue], thread_id: usize) -> Result<TensorValue> {
                if inputs.len() < $count {
                    return Err(anyhow!("cpu kernel expects at least {} inputs", $count));
                }
                $(
                    let $var = $T::from_value(&inputs[$idx])
                        .ok_or_else(|| anyhow!("cpu kernel input {} has wrong dtype", $idx))?;
                )+
                Ok((self)($( &$var.data, )+ thread_id )?.into())
            }
        }
    };
}

macro_rules! impl_cpu_kernel_adapter_attrs {
    ($count:literal, $attr_lt:tt, $(($var:ident, $T:ident, $lt:tt, $idx:tt)),+; $($all_lt:tt),+) => {
        impl<$($T,)+ R> CpuKernelAdapter
            for for<$attr_lt, $($all_lt,)+> fn(&$attr_lt OpAttrs, $( &$lt [$T], )+ usize) -> Result<R>
        where
            $($T: TensorElement + Clone + 'static,)+
            R: Into<TensorValue>,
        {
            fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue], thread_id: usize) -> Result<TensorValue> {
                if inputs.len() < $count {
                    return Err(anyhow!("cpu kernel expects at least {} inputs", $count));
                }
                $(
                    let $var = $T::from_value(&inputs[$idx])
                        .ok_or_else(|| anyhow!("cpu kernel input {} has wrong dtype", $idx))?;
                )+
                Ok((self)(attrs, $( &$var.data, )+ thread_id )?.into())
            }
        }
    };
}

impl_cpu_kernel_adapter_noattrs!(2, (a, A, 'a, 0), (b, B, 'b, 1); 'a, 'b);
impl_cpu_kernel_adapter_noattrs!(3, (a, A, 'a, 0), (b, B, 'b, 1), (c, C, 'c, 2); 'a, 'b, 'c);
impl_cpu_kernel_adapter_noattrs!(
    4,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3);
    'a,
    'b,
    'c,
    'd
);
impl_cpu_kernel_adapter_noattrs!(
    5,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3),
    (e, E, 'e, 4);
    'a,
    'b,
    'c,
    'd,
    'e
);
impl_cpu_kernel_adapter_noattrs!(
    6,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3),
    (e, E, 'e, 4),
    (f, F, 'f, 5);
    'a,
    'b,
    'c,
    'd,
    'e,
    'f
);
impl_cpu_kernel_adapter_noattrs!(
    7,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3),
    (e, E, 'e, 4),
    (f, F, 'f, 5),
    (g, G, 'g, 6);
    'a,
    'b,
    'c,
    'd,
    'e,
    'f,
    'g
);
impl_cpu_kernel_adapter_noattrs!(
    8,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3),
    (e, E, 'e, 4),
    (f, F, 'f, 5),
    (g, G, 'g, 6),
    (h, H, 'h, 7);
    'a,
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h
);
impl_cpu_kernel_adapter_noattrs!(
    9,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3),
    (e, E, 'e, 4),
    (f, F, 'f, 5),
    (g, G, 'g, 6),
    (h, H, 'h, 7),
    (i, I, 'i, 8);
    'a,
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h,
    'i
);
impl_cpu_kernel_adapter_noattrs!(
    10,
    (a, A, 'a, 0),
    (b, B, 'b, 1),
    (c, C, 'c, 2),
    (d, D, 'd, 3),
    (e, E, 'e, 4),
    (f, F, 'f, 5),
    (g, G, 'g, 6),
    (h, H, 'h, 7),
    (i, I, 'i, 8),
    (j, J, 'j, 9);
    'a,
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h,
    'i,
    'j
);

impl_cpu_kernel_adapter_attrs!(2, 'a, (a, A, 'b, 0), (b, B, 'c, 1); 'b, 'c);
impl_cpu_kernel_adapter_attrs!(
    3,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2);
    'b,
    'c,
    'd
);
impl_cpu_kernel_adapter_attrs!(
    4,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3);
    'b,
    'c,
    'd,
    'e
);
impl_cpu_kernel_adapter_attrs!(
    5,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3),
    (e, E, 'f, 4);
    'b,
    'c,
    'd,
    'e,
    'f
);
impl_cpu_kernel_adapter_attrs!(
    6,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3),
    (e, E, 'f, 4),
    (f, F, 'g, 5);
    'b,
    'c,
    'd,
    'e,
    'f,
    'g
);
impl_cpu_kernel_adapter_attrs!(
    7,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3),
    (e, E, 'f, 4),
    (f, F, 'g, 5),
    (g, G, 'h, 6);
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h
);
impl_cpu_kernel_adapter_attrs!(
    8,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3),
    (e, E, 'f, 4),
    (f, F, 'g, 5),
    (g, G, 'h, 6),
    (h, H, 'i, 7);
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h,
    'i
);
impl_cpu_kernel_adapter_attrs!(
    9,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3),
    (e, E, 'f, 4),
    (f, F, 'g, 5),
    (g, G, 'h, 6),
    (h, H, 'i, 7),
    (i, I, 'j, 8);
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h,
    'i,
    'j
);
impl_cpu_kernel_adapter_attrs!(
    10,
    'a,
    (a, A, 'b, 0),
    (b, B, 'c, 1),
    (c, C, 'd, 2),
    (d, D, 'e, 3),
    (e, E, 'f, 4),
    (f, F, 'g, 5),
    (g, G, 'h, 6),
    (h, H, 'i, 7),
    (i, I, 'j, 8),
    (j, J, 'k, 9);
    'b,
    'c,
    'd,
    'e,
    'f,
    'g,
    'h,
    'i,
    'j,
    'k
);

#[cfg(feature = "vulkan")]
impl<R> DeviceKernelAdapter for for<'a> fn(&'a VulkanBuffer, usize) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(
        &self,
        _attrs: &OpAttrs,
        inputs: &[&VulkanBuffer],
        thread_id: usize,
    ) -> Result<VulkanBuffer> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("device kernel expects at least 1 input"))?;
        Ok((self)(a, thread_id)?.into())
    }
}

#[cfg(feature = "vulkan")]
impl<R> DeviceKernelAdapter for for<'a, 'b> fn(&'a OpAttrs, &'b VulkanBuffer, usize) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(
        &self,
        attrs: &OpAttrs,
        inputs: &[&VulkanBuffer],
        thread_id: usize,
    ) -> Result<VulkanBuffer> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("device kernel expects at least 1 input"))?;
        Ok((self)(attrs, a, thread_id)?.into())
    }
}

#[cfg(feature = "vulkan")]
mod device_kernel_adapters {
    use anyhow::{anyhow, Result};

    use crate::backend::VulkanBuffer;
    use crate::graph::OpAttrs;

    use super::DeviceKernelAdapter;

    macro_rules! impl_device_kernel_adapter_noattrs {
        ($count:literal; $($lt:tt),+; $($idx:tt),+) => {
            impl<R> DeviceKernelAdapter for for<$($lt,)+> fn($( &$lt VulkanBuffer, )+ usize) -> Result<R>
            where
                R: Into<VulkanBuffer>,
            {
                fn call(
                    &self,
                    _attrs: &OpAttrs,
                    inputs: &[&VulkanBuffer],
                    thread_id: usize,
                ) -> Result<VulkanBuffer> {
                    if inputs.len() < $count {
                        return Err(anyhow!("device kernel expects at least {} inputs", $count));
                    }
                    Ok((self)($( inputs[$idx], )+ thread_id )?.into())
                }
            }
        };
    }

    macro_rules! impl_device_kernel_adapter_attrs {
        ($count:literal, $attr_lt:tt; $($lt:tt),+; $($idx:tt),+) => {
            impl<R> DeviceKernelAdapter
                for for<$attr_lt, $($lt,)+> fn(&$attr_lt OpAttrs, $( &$lt VulkanBuffer, )+ usize) -> Result<R>
            where
                R: Into<VulkanBuffer>,
            {
                fn call(
                    &self,
                    attrs: &OpAttrs,
                    inputs: &[&VulkanBuffer],
                    thread_id: usize,
                ) -> Result<VulkanBuffer> {
                    if inputs.len() < $count {
                        return Err(anyhow!("device kernel expects at least {} inputs", $count));
                    }
                    Ok((self)(attrs, $( inputs[$idx], )+ thread_id )?.into())
                }
            }
        };
    }

    impl_device_kernel_adapter_noattrs!(2; 'a, 'b; 0, 1);
    impl_device_kernel_adapter_noattrs!(3; 'a, 'b, 'c; 0, 1, 2);
    impl_device_kernel_adapter_noattrs!(4; 'a, 'b, 'c, 'd; 0, 1, 2, 3);
    impl_device_kernel_adapter_noattrs!(5; 'a, 'b, 'c, 'd, 'e; 0, 1, 2, 3, 4);
    impl_device_kernel_adapter_noattrs!(6; 'a, 'b, 'c, 'd, 'e, 'f; 0, 1, 2, 3, 4, 5);
    impl_device_kernel_adapter_noattrs!(
        7;
        'a, 'b, 'c, 'd, 'e, 'f, 'g;
        0, 1, 2, 3, 4, 5, 6
    );
    impl_device_kernel_adapter_noattrs!(
        8;
        'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h;
        0, 1, 2, 3, 4, 5, 6, 7
    );
    impl_device_kernel_adapter_noattrs!(
        9;
        'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i;
        0, 1, 2, 3, 4, 5, 6, 7, 8
    );
    impl_device_kernel_adapter_noattrs!(
        10;
        'a, 'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j;
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    );

    impl_device_kernel_adapter_attrs!(2, 'a; 'b, 'c; 0, 1);
    impl_device_kernel_adapter_attrs!(3, 'a; 'b, 'c, 'd; 0, 1, 2);
    impl_device_kernel_adapter_attrs!(4, 'a; 'b, 'c, 'd, 'e; 0, 1, 2, 3);
    impl_device_kernel_adapter_attrs!(5, 'a; 'b, 'c, 'd, 'e, 'f; 0, 1, 2, 3, 4);
    impl_device_kernel_adapter_attrs!(6, 'a; 'b, 'c, 'd, 'e, 'f, 'g; 0, 1, 2, 3, 4, 5);
    impl_device_kernel_adapter_attrs!(
        7,
        'a;
        'b, 'c, 'd, 'e, 'f, 'g, 'h;
        0, 1, 2, 3, 4, 5, 6
    );
    impl_device_kernel_adapter_attrs!(
        8,
        'a;
        'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i;
        0, 1, 2, 3, 4, 5, 6, 7
    );
    impl_device_kernel_adapter_attrs!(
        9,
        'a;
        'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j;
        0, 1, 2, 3, 4, 5, 6, 7, 8
    );
    impl_device_kernel_adapter_attrs!(
        10,
        'a;
        'b, 'c, 'd, 'e, 'f, 'g, 'h, 'i, 'j, 'k;
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    );
}
