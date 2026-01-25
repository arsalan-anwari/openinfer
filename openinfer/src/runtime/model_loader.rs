use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;

use crate::runtime::tensor_store::{MappedSlice, TensorRef, TensorStore};
use crate::tensor::{
    BF16, Bitset, DType, F16, F8E5M2, I1, I2, I4, T1, T2, U1, U2, U4, Tensor, TensorValue,
};
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
    #[allow(dead_code)]
    path: PathBuf,
    sizes: HashMap<String, usize>,
    vars: HashMap<String, VarInfo>,
    #[allow(dead_code)]
    metadata: HashMap<String, MetadataInfo>,
    mmap: Arc<Mmap>,
    tensor_store: TensorStore,
}

impl ModelLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| "open model file")?;
        let mmap = unsafe { Mmap::map(&file).with_context(|| "mmap model file")? };
        let data = &mmap[..];
        if data.len() < HEADER_SIZE {
            return Err(anyhow!("file too small for OINF header"));
        }

        let mut cursor = 0usize;
        let magic = read_bytes(data, &mut cursor, 5)?;
        if magic != MAGIC {
            return Err(anyhow!("invalid OINF magic"));
        }
        let version = read_u32(data, &mut cursor)?;
        if version != 1 {
            return Err(anyhow!("unsupported OINF version {}", version));
        }
        let _flags = read_u32(data, &mut cursor)?;
        let n_sizevars = read_u32(data, &mut cursor)? as usize;
        let n_metadata = read_u32(data, &mut cursor)? as usize;
        let n_tensors = read_u32(data, &mut cursor)? as usize;
        let _reserved = read_u32(data, &mut cursor)?;
        let offset_sizevars = read_u64(data, &mut cursor)? as usize;
        let offset_metadata = read_u64(data, &mut cursor)? as usize;
        let offset_tensors = read_u64(data, &mut cursor)? as usize;
        let offset_data = read_u64(data, &mut cursor)? as usize;
        let file_size = read_u64(data, &mut cursor)? as usize;

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
            let name = read_string(data, &mut size_cursor)?;
            if sizes.contains_key(&name) {
                return Err(anyhow!("duplicate sizevar {}", name));
            }
            let value = read_u64_at(data, size_cursor)?;
            size_cursor += 8;
            sizes.insert(name, value as usize);
        }

        let mut metadata = HashMap::new();
        let mut meta_cursor = offset_metadata;
        for _ in 0..n_metadata {
            let key = read_string(data, &mut meta_cursor)?;
            if metadata.contains_key(&key) {
                return Err(anyhow!("duplicate metadata key {}", key));
            }
            let value_type = read_u32_at(data, meta_cursor)?;
            let flags = read_u32_at(data, meta_cursor + 4)?;
            let value_nbytes = read_u64_at(data, meta_cursor + 8)?;
            let value_offset = read_u64_at(data, meta_cursor + 16)?;
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
                let element_type = read_u32(data, &mut cursor)?;
                let ndim = read_u32(data, &mut cursor)? as usize;
                if !ValueType::is_scalar(element_type) {
                    return Err(anyhow!("metadata ndarray has invalid element type"));
                }
                for _ in 0..ndim {
                    dims.push(read_u64(data, &mut cursor)?);
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
            let name = read_string(data, &mut tensor_cursor)?;
            if vars.contains_key(&name) {
                return Err(anyhow!("duplicate tensor name {}", name));
            }
            let dtype_raw = read_u32(data, &mut tensor_cursor)?;
            let ndim = read_u32(data, &mut tensor_cursor)? as usize;
            let flags = read_u32(data, &mut tensor_cursor)?;
            let mut dims = Vec::new();
            for _ in 0..ndim {
                dims.push(read_u64(data, &mut tensor_cursor)?);
            }
            let data_nbytes = read_u64(data, &mut tensor_cursor)? as usize;
            let data_offset = read_u64(data, &mut tensor_cursor)? as usize;

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

        let mmap = Arc::new(mmap);
        let tensor_store = build_tensor_store(&sizes, &vars, mmap.clone())?;

        Ok(Self {
            path,
            sizes,
            vars,
            metadata,
            mmap,
            tensor_store,
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

    pub fn tensor_store(&self) -> &TensorStore {
        &self.tensor_store
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
            .ok_or_else(|| anyhow!("missing data range for {}", name))?;
        let data = &self.mmap[range.0..range.1];
        tensor_value_from_bytes(info, data)
    }
}

fn build_tensor_store(
    sizes: &HashMap<String, usize>,
    vars: &HashMap<String, VarInfo>,
    mmap: Arc<Mmap>,
) -> Result<TensorStore> {
    let mut tensors = HashMap::new();
    for (name, info) in vars {
        let shape = resolve_shape(sizes, &info.dims)?;
        let data = info.value_range.map(|(start, end)| {
            MappedSlice::new(mmap.clone(), start..end)
        });
        tensors.insert(
            name.clone(),
            TensorRef {
                name: name.clone(),
                dtype: info.dtype,
                dims: info.dims.clone(),
                shape,
                data,
            },
        );
    }
    Ok(TensorStore::new(tensors))
}

fn resolve_shape(sizes: &HashMap<String, usize>, dims: &[String]) -> Result<Vec<usize>> {
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        shape.push(resolve_dim_value(sizes, dim)?);
    }
    Ok(shape)
}

fn resolve_dim_value(sizes: &HashMap<String, usize>, dim: &str) -> Result<usize> {
    if let Ok(val) = dim.parse::<usize>() {
        return Ok(val);
    }
    let trimmed = dim.trim();
    if let Some((left, right)) = trimmed.split_once('*') {
        let left = left.trim();
        let right = right.trim();
        let left_val = match left.parse::<usize>() {
            Ok(value) => value,
            Err(_) => sizes
                .get(left)
                .copied()
                .ok_or_else(|| anyhow!("unknown size: {}", left))?,
        };
        let right_val = match right.parse::<usize>() {
            Ok(value) => value,
            Err(_) => sizes
                .get(right)
                .copied()
                .ok_or_else(|| anyhow!("unknown size: {}", right))?,
        };
        return Ok(left_val.saturating_mul(right_val));
    }
    sizes
        .get(trimmed)
        .copied()
        .ok_or_else(|| anyhow!("unknown size: {}", trimmed))
}

fn read_bytes<'a>(data: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    if *cursor + len > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    let out = &data[*cursor..*cursor + len];
    *cursor += len;
    Ok(out)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
    let bytes = read_bytes(data, cursor, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64> {
    let bytes = read_bytes(data, cursor, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u32_at(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    Ok(u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()))
}

fn read_u64_at(data: &[u8], offset: usize) -> Result<u64> {
    if offset + 8 > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    Ok(u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()))
}

fn read_string(data: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u32(data, cursor)? as usize;
    let bytes = read_bytes(data, cursor, len)?;
    let s = std::str::from_utf8(bytes).context("invalid UTF-8 string")?;
    let padded = align_up(4 + len, 8);
    let consumed = 4 + len;
    if padded > consumed {
        let skip = padded - consumed;
        if *cursor + skip > data.len() {
            return Err(anyhow!("unexpected EOF"));
        }
        *cursor += skip;
    }
    Ok(s.to_string())
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) / alignment * alignment
}

fn tensor_value_from_bytes(info: &VarInfo, bytes: &[u8]) -> Result<TensorValue> {
    match info.dtype {
        DType::I8 => tensor_from_bytes::<i8>(info, bytes).map(TensorValue::I8),
        DType::I16 => tensor_from_bytes::<i16>(info, bytes).map(TensorValue::I16),
        DType::I32 => tensor_from_bytes::<i32>(info, bytes).map(TensorValue::I32),
        DType::I64 => tensor_from_bytes::<i64>(info, bytes).map(TensorValue::I64),
        DType::U8 => tensor_from_bytes::<u8>(info, bytes).map(TensorValue::U8),
        DType::U16 => tensor_from_bytes::<u16>(info, bytes).map(TensorValue::U16),
        DType::U32 => tensor_from_bytes::<u32>(info, bytes).map(TensorValue::U32),
        DType::U64 => tensor_from_bytes::<u64>(info, bytes).map(TensorValue::U64),
        DType::F16 => tensor_from_bits::<u16, F16>(info, bytes, |bits| F16 { bits }).map(TensorValue::F16),
        DType::BF16 => tensor_from_bits::<u16, BF16>(info, bytes, |bits| BF16 { bits }).map(TensorValue::BF16),
        DType::F8E5M2 => tensor_from_bits::<u8, F8E5M2>(info, bytes, |bits| F8E5M2 { bits }).map(TensorValue::F8E5M2),
        DType::F32 => tensor_from_bytes::<f32>(info, bytes).map(TensorValue::F32),
        DType::F64 => tensor_from_bytes::<f64>(info, bytes).map(TensorValue::F64),
        DType::Bool => tensor_from_bytes::<bool>(info, bytes).map(TensorValue::Bool),
        DType::Bitset => tensor_from_bits::<u8, Bitset>(info, bytes, |bits| Bitset { bits }).map(TensorValue::Bitset),
        DType::I4 => tensor_from_bits::<u8, I4>(info, bytes, |bits| I4 { bits }).map(TensorValue::I4),
        DType::I2 => tensor_from_bits::<u8, I2>(info, bytes, |bits| I2 { bits }).map(TensorValue::I2),
        DType::I1 => tensor_from_bits::<u8, I1>(info, bytes, |bits| I1 { bits }).map(TensorValue::I1),
        DType::U4 => tensor_from_bits::<u8, U4>(info, bytes, |bits| U4 { bits }).map(TensorValue::U4),
        DType::U2 => tensor_from_bits::<u8, U2>(info, bytes, |bits| U2 { bits }).map(TensorValue::U2),
        DType::U1 => tensor_from_bits::<u8, U1>(info, bytes, |bits| U1 { bits }).map(TensorValue::U1),
        DType::T2 => tensor_from_bits::<u8, T2>(info, bytes, |bits| T2 { bits }).map(TensorValue::T2),
        DType::T1 => tensor_from_bits::<u8, T1>(info, bytes, |bits| T1 { bits }).map(TensorValue::T1),
    }
}

fn tensor_from_bytes<T: Copy>(info: &VarInfo, bytes: &[u8]) -> Result<Tensor<T>> {
    let shape = info
        .dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("invalid tensor dims for {}", info.name))?;
    let len = shape.iter().product::<usize>();
    let expected = len * std::mem::size_of::<T>();
    if bytes.len() != expected {
        return Err(anyhow!(
            "tensor {} byte length mismatch: expected {}, got {}",
            info.name,
            expected,
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(len);
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let end = cursor + std::mem::size_of::<T>();
        let value = read_t::<T>(&bytes[cursor..end])?;
        out.push(value);
        cursor = end;
    }
    Tensor::from_vec_with_opts(
        out,
        crate::tensor::TensorOptions {
            shape: Some(shape),
            ..crate::tensor::TensorOptions::default()
        },
    )
}

fn tensor_from_bits<B: Copy, T>(
    info: &VarInfo,
    bytes: &[u8],
    map: fn(B) -> T,
) -> Result<Tensor<T>> {
    let shape = info
        .dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("invalid tensor dims for {}", info.name))?;
    let len = shape.iter().product::<usize>();
    if bytes.is_empty() && len == 0 {
        return Tensor::from_vec_with_opts(
            Vec::new(),
            crate::tensor::TensorOptions {
                shape: Some(shape),
                ..crate::tensor::TensorOptions::default()
            },
        );
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let end = cursor + std::mem::size_of::<B>();
        let value = read_t::<B>(&bytes[cursor..end])?;
        out.push(map(value));
        cursor = end;
    }
    Tensor::from_vec_with_opts(
        out,
        crate::tensor::TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..crate::tensor::TensorOptions::default()
        },
    )
}

fn read_t<T: Copy>(bytes: &[u8]) -> Result<T> {
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    let len = std::mem::size_of::<T>();
    if bytes.len() != len {
        return Err(anyhow!("invalid byte length"));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.as_mut_ptr() as *mut u8, len);
        Ok(value.assume_init())
    }
}

struct ValueType;

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
    #[allow(dead_code)]
    const STRING: u32 = 14;
    const NDARRAY: u32 = 15;
    const BF16: u32 = 16;
    const F8E5M2: u32 = 17;
    const I4: u32 = 18;
    const I2: u32 = 19;
    const I1: u32 = 20;
    const U4: u32 = 21;
    const U2: u32 = 22;
    const U1: u32 = 23;
    const T2: u32 = 24;
    const T1: u32 = 25;

    fn is_scalar(value_type: u32) -> bool {
        value_type >= Self::I8 && value_type <= Self::T1
    }

    fn to_dtype(value_type: u32) -> Result<DType> {
        Ok(match value_type {
            Self::I8 => DType::I8,
            Self::I16 => DType::I16,
            Self::I32 => DType::I32,
            Self::I64 => DType::I64,
            Self::U8 => DType::U8,
            Self::U16 => DType::U16,
            Self::U32 => DType::U32,
            Self::U64 => DType::U64,
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
            Self::F64 => DType::F64,
            Self::BOOL => DType::Bool,
            Self::BITSET => DType::Bitset,
            Self::BF16 => DType::BF16,
            Self::F8E5M2 => DType::F8E5M2,
            Self::I4 => DType::I4,
            Self::I2 => DType::I2,
            Self::I1 => DType::I1,
            Self::U4 => DType::U4,
            Self::U2 => DType::U2,
            Self::U1 => DType::U1,
            Self::T2 => DType::T2,
            Self::T1 => DType::T1,
            _ => return Err(anyhow!("unknown tensor dtype {}", value_type)),
        })
    }
}
