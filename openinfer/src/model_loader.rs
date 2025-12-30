use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::tensor::{DType, Tensor, TensorValue};
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
            if let Ok(val) = dim.parse::<usize>() {
                total = total.saturating_mul(val);
            } else {
                let v = self.size_of(dim)?;
                total = total.saturating_mul(v);
            }
        }
        Ok(total)
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
        let dtype_size = dtype_nbytes(info.dtype);
        if expected_len * dtype_size != nbytes {
            return Err(anyhow!(
                "tensor {} size mismatch: expected {} bytes, got {}",
                name,
                expected_len * dtype_size,
                nbytes
            ));
        }

        match info.dtype {
            DType::F32 => Ok(TensorValue::F32(Tensor::new(parse_f32(&buf)?))),
            DType::F64 => Ok(TensorValue::F64(Tensor::new(parse_f64(&buf)?))),
            DType::I32 => Ok(TensorValue::I32(Tensor::new(parse_i32(&buf)?))),
            DType::I64 => Ok(TensorValue::I64(Tensor::new(parse_i64(&buf)?))),
            DType::Bool => Ok(TensorValue::Bool(Tensor::new(parse_bool(&buf)?))),
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
        )
    }

    fn to_dtype(value_type: u32) -> Result<DType> {
        match value_type {
            Self::F32 => Ok(DType::F32),
            Self::F64 => Ok(DType::F64),
            Self::I32 => Ok(DType::I32),
            Self::I64 => Ok(DType::I64),
            Self::BOOL => Ok(DType::Bool),
            other => Err(anyhow!("unsupported tensor dtype {}", other)),
        }
    }
}

fn dtype_nbytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::I32 => 4,
        DType::I64 => 8,
        DType::Bool => 1,
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

fn parse_bool(buf: &[u8]) -> Result<Vec<bool>> {
    Ok(buf.iter().map(|b| *b != 0).collect())
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
