use anyhow::{anyhow, Result};

use crate::tensor::{DType, TensorValue};

#[allow(dead_code)]
pub fn tensor_to_bytes(value: &TensorValue) -> Result<Vec<u8>> {
    Ok(match value {
        TensorValue::F32(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::F64(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I8(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I16(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I32(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I64(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::U8(tensor) => tensor.data.clone(),
        TensorValue::U16(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::U32(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::U64(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::Bool(tensor) => tensor.data.iter().map(|v| if *v { 1 } else { 0 }).collect(),
        TensorValue::Bitset(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::F16(tensor) => tensor.data.iter().flat_map(|v| v.bits.to_le_bytes()).collect(),
        TensorValue::BF16(tensor) => tensor.data.iter().flat_map(|v| v.bits.to_le_bytes()).collect(),
        TensorValue::F8E5M2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I4(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U4(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::T1(_) | TensorValue::T2(_) => {
            return Err(anyhow!("ternary packed types not supported in vulkan"))
        }
    })
}

pub fn tensor_append_bytes(value: &TensorValue, out: &mut Vec<u8>) -> Result<()> {
    match value {
        TensorValue::F32(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::F64(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I8(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I16(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I32(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I64(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::U8(tensor) => out.extend_from_slice(&tensor.data),
        TensorValue::U16(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::U32(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::U64(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::Bool(tensor) => out.extend(tensor.data.iter().map(|v| if *v { 1 } else { 0 })),
        TensorValue::Bitset(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::F16(tensor) => out.extend(tensor.data.iter().flat_map(|v| v.bits.to_le_bytes())),
        TensorValue::BF16(tensor) => out.extend(tensor.data.iter().flat_map(|v| v.bits.to_le_bytes())),
        TensorValue::F8E5M2(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::I4(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::I2(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::I1(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::U4(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::U2(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::U1(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::T1(_) | TensorValue::T2(_) => {
            return Err(anyhow!("ternary packed types not supported in vulkan"))
        }
    }
    Ok(())
}

pub fn write_tensor_from_bytes(output: &mut TensorValue, bytes: &[u8]) -> Result<()> {
    match output {
        TensorValue::F32(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::F64(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I8(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I16(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I32(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I64(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::U8(tensor) => {
            tensor.data = bytes.to_vec();
        }
        TensorValue::U16(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::U32(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::U64(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::Bool(tensor) => {
            tensor.data = bytes.iter().map(|v| *v != 0).collect();
        }
        TensorValue::Bitset(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::Bitset { bits: *v }).collect();
        }
        TensorValue::F16(tensor) => {
            let mut out = Vec::new();
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(crate::tensor::F16 { bits });
            }
            tensor.data = out;
        }
        TensorValue::BF16(tensor) => {
            let mut out = Vec::new();
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(crate::tensor::BF16 { bits });
            }
            tensor.data = out;
        }
        TensorValue::F8E5M2(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::F8E5M2 { bits: *v }).collect();
        }
        TensorValue::I4(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::I4 { bits: *v }).collect();
        }
        TensorValue::I2(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::I2 { bits: *v }).collect();
        }
        TensorValue::I1(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::I1 { bits: *v }).collect();
        }
        TensorValue::U4(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::U4 { bits: *v }).collect();
        }
        TensorValue::U2(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::U2 { bits: *v }).collect();
        }
        TensorValue::U1(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::U1 { bits: *v }).collect();
        }
        TensorValue::T1(_) | TensorValue::T2(_) => {
            return Err(anyhow!("ternary packed types not supported in vulkan"))
        }
    }
    Ok(())
}

pub fn tensor_byte_len(dtype: DType, logical_len: usize) -> usize {
    if dtype.is_packed() {
        dtype.storage_len(logical_len)
    } else {
        let elem_bytes = (dtype.bit_width() as usize + 7) / 8;
        logical_len.saturating_mul(elem_bytes)
    }
}
