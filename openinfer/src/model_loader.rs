use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::tensor::{BF16, Bitset, DType, F16, F8E5M2, I1, I2, I4, Tensor, TensorValue};
use crate::types::VarInfo;

const MAGIC: &[u8; 5] = b"OINF\0";
const HEADER_SIZE: usize = 69;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MetadataInfo {
    value_type: u32,
    value_offset: u64,
    value_nbytes: u64,
    dims: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct ModelLoader {
    path: PathBuf,
    sizes: HashMap<String, usize>,
    vars: HashMap<String, VarInfo>,
    #[allow(dead_code)]
    metadata: HashMap<String, MetadataInfo>,
}

impl ModelLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let data = fs::read(&path).with_context(|| "read model file")?;
        if data.len() < HEADER_SIZE {
            return Err(anyhow!("file too small for OINF header"));
        }

        let mut cursor = 0usize;
        let magic = read_bytes(&data, &mut cursor, 5)?;
        if magic != MAGIC {
            return Err(anyhow!("invalid OINF magic"));
        }
        let version = read_u32(&data, &mut cursor)?;
        if version != 1 {
            return Err(anyhow!("unsupported OINF version {}", version));
        }
        let _flags = read_u32(&data, &mut cursor)?;
        let n_sizevars = read_u32(&data, &mut cursor)? as usize;
        let n_metadata = read_u32(&data, &mut cursor)? as usize;
        let n_tensors = read_u32(&data, &mut cursor)? as usize;
        let _reserved = read_u32(&data, &mut cursor)?;
        let offset_sizevars = read_u64(&data, &mut cursor)? as usize;
        let offset_metadata = read_u64(&data, &mut cursor)? as usize;
        let offset_tensors = read_u64(&data, &mut cursor)? as usize;
        let offset_data = read_u64(&data, &mut cursor)? as usize;
        let file_size = read_u64(&data, &mut cursor)? as usize;

        if file_size != data.len() {
            return Err(anyhow!("file size mismatch"));
        }
        let offsets = vec![
            offset_sizevars,
            offset_metadata,
            offset_tensors,
            offset_data,
            file_size,
        ];
        let mut sorted = offsets.clone();
        sorted.sort_unstable();
        if offsets != sorted {
            return Err(anyhow!("OINF offsets are not ascending"));
        }
        for off in offsets.iter().take(4) {
            if *off % 8 != 0 {
                return Err(anyhow!("OINF section offset not aligned"));
            }
            if *off > file_size {
                return Err(anyhow!("OINF section offset out of bounds"));
            }
        }

        let mut sizes = HashMap::new();
        let mut size_cursor = offset_sizevars;
        for _ in 0..n_sizevars {
            let name = read_string(&data, &mut size_cursor)?;
            if sizes.contains_key(&name) {
                return Err(anyhow!("duplicate sizevar {}", name));
            }
            let value = read_u64_at(&data, size_cursor)?;
            size_cursor += 8;
            sizes.insert(name, value as usize);
        }

        let mut metadata = HashMap::new();
        let mut meta_cursor = offset_metadata;
        for _ in 0..n_metadata {
            let key = read_string(&data, &mut meta_cursor)?;
            if metadata.contains_key(&key) {
                return Err(anyhow!("duplicate metadata key {}", key));
            }
            let value_type = read_u32_at(&data, meta_cursor)?;
            let flags = read_u32_at(&data, meta_cursor + 4)?;
            let value_nbytes = read_u64_at(&data, meta_cursor + 8)?;
            let value_offset = read_u64_at(&data, meta_cursor + 16)?;
            meta_cursor += 24;
            if flags != 0 {
                return Err(anyhow!("metadata flags must be 0"));
            }
            if value_offset % 8 != 0 {
                return Err(anyhow!("metadata value offset not aligned"));
            }
            let value_end = value_offset
                .checked_add(value_nbytes)
                .ok_or_else(|| anyhow!("metadata value offset overflow"))?;
            if value_end as usize > file_size {
                return Err(anyhow!("metadata value out of bounds"));
            }

            let mut dims = Vec::new();
            if value_type == ValueType::NDARRAY {
                let mut cursor = value_offset as usize;
                let element_type = read_u32(&data, &mut cursor)?;
                let ndim = read_u32(&data, &mut cursor)? as usize;
                if !ValueType::is_scalar(element_type) {
                    return Err(anyhow!("metadata ndarray has invalid element type"));
                }
                for _ in 0..ndim {
                    dims.push(read_u64(&data, &mut cursor)?);
                }
            }

            metadata.insert(
                key,
                MetadataInfo {
                    value_type,
                    value_offset,
                    value_nbytes,
                    dims,
                },
            );
        }

        let mut vars = HashMap::new();
        let mut tensor_cursor = offset_tensors;
        for _ in 0..n_tensors {
            let name = read_string(&data, &mut tensor_cursor)?;
            if vars.contains_key(&name) {
                return Err(anyhow!("duplicate tensor name {}", name));
            }
            let dtype_raw = read_u32(&data, &mut tensor_cursor)?;
            let ndim = read_u32(&data, &mut tensor_cursor)? as usize;
            let flags = read_u32(&data, &mut tensor_cursor)?;
            let mut dims = Vec::new();
            for _ in 0..ndim {
                dims.push(read_u64(&data, &mut tensor_cursor)?);
            }
            let data_nbytes = read_u64(&data, &mut tensor_cursor)? as usize;
            let data_offset = read_u64(&data, &mut tensor_cursor)? as usize;

            let dtype = ValueType::to_dtype(dtype_raw)?;
            let has_data = (flags & 1) != 0;
            if has_data {
                if data_offset % 8 != 0 {
                    return Err(anyhow!("tensor data offset not aligned"));
                }
                if data_offset < offset_data {
                    return Err(anyhow!("tensor data offset precedes data section"));
                }
                if data_offset + data_nbytes > file_size {
                    return Err(anyhow!("tensor data out of bounds"));
                }
            } else if data_offset != 0 || data_nbytes != 0 {
                return Err(anyhow!("tensor without data must have zero offset/size"));
            }

            let dims_str = dims.iter().map(|d| d.to_string()).collect();
            let value_range = if has_data {
                Some((data_offset, data_offset + data_nbytes))
            } else {
                None
            };
            vars.insert(
                name.clone(),
                VarInfo {
                    name,
                    dtype,
                    dims: dims_str,
                    value_range,
                    has_data,
                },
            );
        }

        Ok(Self {
            path,
            sizes,
            vars,
            metadata,
        })
    }

    pub fn size_of(&self, name: &str) -> Result<usize> {
        self.sizes
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("unknown size: {}", name))
    }

    pub fn resolve_len(&self, dims: &[String]) -> Result<usize> {
        let mut total = 1usize;
        for dim in dims {
            total = total.saturating_mul(self.resolve_dim_value(dim)?);
        }
        Ok(total)
    }

    pub fn resolve_shape(&self, dims: &[String]) -> Result<Vec<usize>> {
        let mut shape = Vec::with_capacity(dims.len());
        for dim in dims {
            shape.push(self.resolve_dim_value(dim)?);
        }
        Ok(shape)
    }

    pub fn resolve_dim_value(&self, dim: &str) -> Result<usize> {
        if let Ok(val) = dim.parse::<usize>() {
            return Ok(val);
        }
        let trimmed = dim.trim();
        if let Some((left, right)) = trimmed.split_once('*') {
            let left = left.trim();
            let right = right.trim();
            let left_val = match left.parse::<usize>() {
                Ok(value) => value,
                Err(_) => self.size_of(left)?,
            };
            let right_val = match right.parse::<usize>() {
                Ok(value) => value,
                Err(_) => self.size_of(right)?,
            };
            return Ok(left_val.saturating_mul(right_val));
        }
        self.size_of(trimmed)
    }

    pub fn var_info(&self, name: &str) -> Option<&VarInfo> {
        self.vars.get(name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<TensorValue> {
        let info = self
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
        if !info.has_data {
            return Err(anyhow!("no data found for {}", name));
        }
        let range = info
            .value_range
            .ok_or_else(|| anyhow!("no data range for {}", name))?;
        let nbytes = range.1 - range.0;

        let mut file = File::open(&self.path).with_context(|| "open model file")?;
        file.seek(SeekFrom::Start(range.0 as u64))
            .with_context(|| "seek tensor data")?;
        let mut buf = vec![0u8; nbytes];
        file.read_exact(&mut buf)
            .with_context(|| "read tensor data")?;

        let expected_len = self
            .resolve_len(&info.dims)
            .with_context(|| "resolve tensor length")?;
        let expected_bytes = expected_nbytes(info.dtype, expected_len);
        if expected_bytes != nbytes {
            return Err(anyhow!(
                "tensor {} size mismatch: expected {} bytes, got {}",
                name, expected_bytes, nbytes
            ));
        }

        let shape = self.resolve_shape(&info.dims)?;
        match info.dtype {
            DType::I8 => Ok(TensorValue::I8(Tensor::from_vec_with_opts(parse_i8(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::I16 => Ok(TensorValue::I16(Tensor::from_vec_with_opts(parse_i16(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::F32 => Ok(TensorValue::F32(Tensor::from_vec_with_opts(parse_f32(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::F64 => Ok(TensorValue::F64(Tensor::from_vec_with_opts(parse_f64(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::U8 => Ok(TensorValue::U8(Tensor::from_vec_with_opts(parse_u8(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::U16 => Ok(TensorValue::U16(Tensor::from_vec_with_opts(parse_u16(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::I32 => Ok(TensorValue::I32(Tensor::from_vec_with_opts(parse_i32(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::I64 => Ok(TensorValue::I64(Tensor::from_vec_with_opts(parse_i64(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::U32 => Ok(TensorValue::U32(Tensor::from_vec_with_opts(parse_u32(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::U64 => Ok(TensorValue::U64(Tensor::from_vec_with_opts(parse_u64(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::Bool => Ok(TensorValue::Bool(Tensor::from_vec_with_opts(parse_bool(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::Bitset => Ok(TensorValue::Bitset(Tensor::from_vec_with_opts(parse_bitset(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::F16 => Ok(TensorValue::F16(Tensor::from_vec_with_opts(parse_f16(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::BF16 => Ok(TensorValue::BF16(Tensor::from_vec_with_opts(parse_bf16(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::F8E5M2 => Ok(TensorValue::F8E5M2(Tensor::from_vec_with_opts(parse_f8(&buf)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::I4 => Ok(TensorValue::I4(Tensor::from_vec_with_opts(parse_i4(&buf, expected_len)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::I2 => Ok(TensorValue::I2(Tensor::from_vec_with_opts(parse_i2(&buf, expected_len)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
            DType::I1 => Ok(TensorValue::I1(Tensor::from_vec_with_opts(parse_i1(&buf, expected_len)?, crate::tensor::TensorOptions {
                shape: Some(shape.clone()),
                ..crate::tensor::TensorOptions::default()
            })?)),
        }
    }

}

struct ValueType;

#[allow(dead_code)]
impl ValueType {
    const I8: u32 = 1;
    const I16: u32 = 2;
    const I32: u32 = 3;
    const I64: u32 = 4;
    const U8: u32 = 5;
    const U16: u32 = 6;
    const U32: u32 = 7;
    const U64: u32 = 8;
    const F16: u32 = 9;
    const F32: u32 = 10;
    const F64: u32 = 11;
    const BOOL: u32 = 12;
    const BITSET: u32 = 13;
    const STRING: u32 = 14;
    const NDARRAY: u32 = 15;
    const BF16: u32 = 16;
    const F8E5M2: u32 = 17;
    const I4: u32 = 18;
    const I2: u32 = 19;
    const I1: u32 = 20;

    fn is_scalar(value_type: u32) -> bool {
        matches!(
            value_type,
            Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::F16
                | Self::F32
                | Self::F64
                | Self::BOOL
                | Self::BITSET
                | Self::BF16
                | Self::F8E5M2
                | Self::I4
                | Self::I2
                | Self::I1
        )
    }

    fn to_dtype(value_type: u32) -> Result<DType> {
        match value_type {
            Self::I8 => Ok(DType::I8),
            Self::I16 => Ok(DType::I16),
            Self::F32 => Ok(DType::F32),
            Self::F64 => Ok(DType::F64),
            Self::U8 => Ok(DType::U8),
            Self::U16 => Ok(DType::U16),
            Self::I32 => Ok(DType::I32),
            Self::I64 => Ok(DType::I64),
            Self::U32 => Ok(DType::U32),
            Self::U64 => Ok(DType::U64),
            Self::BOOL => Ok(DType::Bool),
            Self::BITSET => Ok(DType::Bitset),
            Self::F16 => Ok(DType::F16),
            Self::BF16 => Ok(DType::BF16),
            Self::F8E5M2 => Ok(DType::F8E5M2),
            Self::I4 => Ok(DType::I4),
            Self::I2 => Ok(DType::I2),
            Self::I1 => Ok(DType::I1),
            other => Err(anyhow!("unsupported tensor dtype {}", other)),
        }
    }
}

fn expected_nbytes(dtype: DType, len: usize) -> usize {
    match dtype {
        DType::I1 => (len + 7) / 8,
        DType::I2 => (len + 3) / 4,
        DType::I4 => (len + 1) / 2,
        DType::F8E5M2 => len,
        DType::BF16 => len * 2,
        DType::F16 => len * 2,
        DType::I8 | DType::U8 | DType::Bool | DType::Bitset => len,
        DType::I16 | DType::U16 => len * 2,
        DType::I32 | DType::U32 | DType::F32 => len * 4,
        DType::I64 | DType::U64 | DType::F64 => len * 8,
    }
}

fn parse_f32(buf: &[u8]) -> Result<Vec<f32>> {
    if buf.len() % 4 != 0 {
        return Err(anyhow!("f32 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_i8(buf: &[u8]) -> Result<Vec<i8>> {
    Ok(buf.iter().map(|b| *b as i8).collect())
}

fn parse_i16(buf: &[u8]) -> Result<Vec<i16>> {
    if buf.len() % 2 != 0 {
        return Err(anyhow!("i16 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_u8(buf: &[u8]) -> Result<Vec<u8>> {
    Ok(buf.to_vec())
}

fn parse_u16(buf: &[u8]) -> Result<Vec<u16>> {
    if buf.len() % 2 != 0 {
        return Err(anyhow!("u16 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_f64(buf: &[u8]) -> Result<Vec<f64>> {
    if buf.len() % 8 != 0 {
        return Err(anyhow!("f64 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_i32(buf: &[u8]) -> Result<Vec<i32>> {
    if buf.len() % 4 != 0 {
        return Err(anyhow!("i32 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_i64(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.len() % 8 != 0 {
        return Err(anyhow!("i64 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_u32(buf: &[u8]) -> Result<Vec<u32>> {
    if buf.len() % 4 != 0 {
        return Err(anyhow!("u32 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_u64(buf: &[u8]) -> Result<Vec<u64>> {
    if buf.len() % 8 != 0 {
        return Err(anyhow!("u64 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_bool(buf: &[u8]) -> Result<Vec<bool>> {
    Ok(buf.iter().map(|b| *b != 0).collect())
}

fn parse_bitset(buf: &[u8]) -> Result<Vec<Bitset>> {
    Ok(buf.iter().map(|b| Bitset { bits: *b }).collect())
}

fn parse_f16(buf: &[u8]) -> Result<Vec<F16>> {
    if buf.len() % 2 != 0 {
        return Err(anyhow!("f16 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(2)
        .map(|chunk| F16 {
            bits: u16::from_le_bytes(chunk.try_into().unwrap()),
        })
        .collect())
}

fn parse_bf16(buf: &[u8]) -> Result<Vec<BF16>> {
    if buf.len() % 2 != 0 {
        return Err(anyhow!("bf16 tensor bytes not aligned"));
    }
    Ok(buf
        .chunks_exact(2)
        .map(|chunk| BF16 {
            bits: u16::from_le_bytes(chunk.try_into().unwrap()),
        })
        .collect())
}

fn parse_f8(buf: &[u8]) -> Result<Vec<F8E5M2>> {
    Ok(buf.iter().map(|b| F8E5M2 { bits: *b }).collect())
}

fn unpack_packed_bits(buf: &[u8], bits_per: u8, len: usize) -> Result<Vec<u8>> {
    if bits_per == 0 || bits_per > 8 {
        return Err(anyhow!("invalid packed bit width {}", bits_per));
    }
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let bit_index = idx * bits_per as usize;
        let byte_index = bit_index / 8;
        let shift = (bit_index % 8) as u8;
        if byte_index >= buf.len() {
            return Err(anyhow!("packed tensor data out of bounds"));
        }
        let mut value = (buf[byte_index] >> shift) as u16;
        let remaining = 8u8.saturating_sub(shift);
        if remaining < bits_per {
            if byte_index + 1 >= buf.len() {
                return Err(anyhow!("packed tensor data out of bounds"));
            }
            value |= (buf[byte_index + 1] as u16) << remaining;
        }
        let mask = (1u16 << bits_per) - 1;
        out.push((value & mask) as u8);
    }
    Ok(out)
}

fn parse_i4(buf: &[u8], len: usize) -> Result<Vec<I4>> {
    let values = unpack_packed_bits(buf, 4, len)?;
    Ok(values.into_iter().map(|bits| I4 { bits }).collect())
}

fn parse_i2(buf: &[u8], len: usize) -> Result<Vec<I2>> {
    let values = unpack_packed_bits(buf, 2, len)?;
    Ok(values.into_iter().map(|bits| I2 { bits }).collect())
}

fn parse_i1(buf: &[u8], len: usize) -> Result<Vec<I1>> {
    let values = unpack_packed_bits(buf, 1, len)?;
    Ok(values.into_iter().map(|bits| I1 { bits }).collect())
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
    let value = read_u32_at(data, *cursor)?;
    *cursor += 4;
    Ok(value)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64> {
    let value = read_u64_at(data, *cursor)?;
    *cursor += 8;
    Ok(value)
}

fn read_u32_at(data: &[u8], offset: usize) -> Result<u32> {
    let end = offset + 4;
    if end > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    Ok(u32::from_le_bytes(data[offset..end].try_into().unwrap()))
}

fn read_u64_at(data: &[u8], offset: usize) -> Result<u64> {
    let end = offset + 8;
    if end > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    Ok(u64::from_le_bytes(data[offset..end].try_into().unwrap()))
}

fn read_bytes<'a>(data: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = *cursor + len;
    if end > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    let out = &data[*cursor..end];
    *cursor = end;
    Ok(out)
}

fn read_string(data: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u32(data, cursor)? as usize;
    let bytes = read_bytes(data, cursor, len)?;
    let text = std::str::from_utf8(bytes).map_err(|_| anyhow!("invalid ASCII string"))?;
    if !text.chars().all(is_key_char) {
        return Err(anyhow!("invalid key '{}'", text));
    }
    let padded = align_up(4 + len);
    *cursor += padded - (4 + len);
    Ok(text.to_string())
}

fn is_key_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-'
}

fn align_up(value: usize) -> usize {
    (value + 7) / 8 * 8
}
