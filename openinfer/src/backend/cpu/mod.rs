use crate::backend::TensorStorage;
use crate::graph::{OpAttrs, OpKind};
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{DType, TensorValue};
use anyhow::Result;

pub mod scheduler;

impl DeviceBackend for Device {
    fn device(&self) -> Device {
        *self
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<TensorStorage> {
        Ok(TensorStorage::Host(TensorValue::zeros(dtype, shape)))
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        Ok(TensorStorage::Host(value))
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        match value {
            TensorStorage::Host(host) => Ok(host),
        }
    }

    fn exec_op(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[&TensorStorage],
        output: &mut TensorStorage,
        thread_id: usize,
        is_inplace: bool,
    ) -> Result<()> {
        log_exec_op(
            *self,
            op,
            attrs,
            output_dtype,
            tensors,
            output,
            thread_id,
            is_inplace,
        );
        Ok(())
    }

    fn exec_op_inplace(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        output: &mut TensorStorage,
        inputs: &[&TensorStorage],
        thread_id: usize,
    ) -> Result<()> {
        let mut tensors = Vec::with_capacity(inputs.len() + 1);
        tensors.push(output as &TensorStorage);
        tensors.extend_from_slice(inputs);
        log_exec_op(
            *self,
            op,
            attrs,
            output_dtype,
            &tensors,
            output,
            thread_id,
            true,
        );
        Ok(())
    }
}

fn log_exec_op(
    device: Device,
    op: OpKind,
    attrs: &OpAttrs,
    output_dtype: DType,
    tensors: &[&TensorStorage],
    output: &TensorStorage,
    thread_id: usize,
    is_inplace: bool,
) {
    let _ = thread_id;
    let is_accumulate = matches!(attrs, OpAttrs::Accumulate { .. });
    let input_summary = format_tensor_list(tensors);
    let output_summary = format_tensor(output);
    let is_broadcast = compute_broadcast(tensors);
    println!(
        "exec_op\n\t- device: {:?}\n\t- op: {}\n\t- inputs: {}\n\t- attrs: {:?}\n\t- output_dtype: {:?}\n\t- outputs: {}\n\t- is_inplace: {}\n\t- is_broadcast: {}\n\t- is_accumulate: {}",
        device,
        op.as_str(),
        input_summary,
        attrs,
        output_dtype,
        output_summary,
        is_inplace,
        is_broadcast,
        is_accumulate
    );
}

fn format_tensor(storage: &TensorStorage) -> String {
    match storage {
        TensorStorage::Host(value) => format!(
            "Host(dtype={:?}, shape={:?})",
            value.dtype(),
            value.shape()
        ),
    }
}

fn format_tensor_list(tensors: &[&TensorStorage]) -> String {
    if tensors.is_empty() {
        return "[]".to_string();
    }
    let formatted = tensors
        .iter()
        .map(|tensor| format_tensor(*tensor))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", formatted)
}

fn compute_broadcast(tensors: &[&TensorStorage]) -> bool {
    let mut base_shape: Option<Vec<usize>> = None;
    for tensor in tensors {
        let shape = match tensor {
            TensorStorage::Host(value) => value.shape().to_vec(),
        };
        if let Some(base) = &base_shape {
            if *base != shape {
                return true;
            }
        } else {
            base_shape = Some(shape);
        }
    }
    false
}
