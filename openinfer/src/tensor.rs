use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::cell::UnsafeCell;
use std::ops::Index;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bitset {
    pub bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct F16 {
    pub bits: u16,
}

impl F16 {
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7fffff;
        let f16_bits = match exp {
            0 => sign,
            255 => {
                if mant == 0 {
                    sign | 0x7c00
                } else {
                    sign | 0x7c00 | ((mant >> 13) as u16) | 1
                }
            }
            _ => {
                let exp16 = exp - 127 + 15;
                if exp16 >= 0x1f {
                    sign | 0x7c00
                } else if exp16 <= 0 {
                    if exp16 < -10 {
                        sign
                    } else {
                        let mant16 = mant | 0x800000;
                        let shift = (14 - exp16) as u32;
                        let mut half = (mant16 >> shift) as u16;
                        if (mant16 >> (shift - 1)) & 1 != 0 {
                            half = half.wrapping_add(1);
                        }
                        sign | half
                    }
                } else {
                    let mut half = ((exp16 as u16) << 10) | ((mant >> 13) as u16);
                    if (mant >> 12) & 1 != 0 {
                        half = half.wrapping_add(1);
                    }
                    sign | half
                }
            }
        };
        Self { bits: f16_bits }
    }

    pub fn to_f32(self) -> f32 {
        let sign = ((self.bits & 0x8000) as u32) << 16;
        let exp = (self.bits >> 10) & 0x1f;
        let mant = (self.bits & 0x03ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut mant = mant;
                let mut exp = -1i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    exp -= 1;
                }
                mant &= 0x03ff;
                let exp32 = (exp + 1 + 127 - 15) as u32;
                sign | (exp32 << 23) | (mant << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f800000 | (mant << 13)
        } else {
            let exp32 = (exp as u32) + (127 - 15);
            sign | (exp32 << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TensorOptions {
    pub shape: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct TensorView<T> {
    data: *const T,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> TensorView<T> {
    fn new(data: *const T, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        numel(&self.shape)
    }

    pub fn at(&self, indices: &[usize]) -> &T {
        let offset = offset_for(&self.shape, &self.strides, indices)
            .unwrap_or_else(|err| panic!("tensor view index error: {}", err));
        unsafe { &*self.data.add(offset) }
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        if !is_contiguous(&self.shape, &self.strides) {
            return None;
        }
        let len = self.len();
        if len == 0 {
            return Some(&[]);
        }
        unsafe { Some(std::slice::from_raw_parts(self.data, len)) }
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if let Some(slice) = self.as_slice() {
            return slice.to_vec();
        }
        let mut out = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            let coords = linear_to_indices(idx, &self.shape);
            out.push(self.at(&coords).clone());
        }
        out
    }
}

#[derive(Debug)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    // Indexing caches a view; this is not thread-safe.
    view_cache: UnsafeCell<TensorView<T>>,
}

// Tensor owns its backing storage; moving between threads is safe when it is
// not accessed concurrently.
unsafe impl<T: Send> Send for Tensor<T> {}

impl<T: Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        let data = self.data.clone();
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let data_ptr = data.as_ptr();
        Self {
            data,
            shape: shape.clone(),
            strides: strides.clone(),
            view_cache: UnsafeCell::new(TensorView::new(data_ptr, shape, strides)),
        }
    }
}

impl<T> Tensor<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Self> {
        Self::from_vec_with_opts(data, TensorOptions::default())
    }

    pub fn from_vec_with_opts(data: Vec<T>, opts: TensorOptions) -> Result<Self> {
        let shape = match opts.shape {
            Some(shape) => shape,
            None => vec![data.len()],
        };
        let expected = numel(&shape);
        if expected != data.len() {
            return Err(anyhow!(
                "tensor shape {:?} expects {} values, got {}",
                shape,
                expected,
                data.len()
            ));
        }
        if shape.is_empty() && data.len() != 1 {
            return Err(anyhow!(
                "scalar tensor expects 1 value, got {}",
                data.len()
            ));
        }
        let strides = match opts.strides {
            Some(strides) => {
                if strides.len() != shape.len() {
                    return Err(anyhow!(
                        "tensor strides length {} does not match shape length {}",
                        strides.len(),
                        shape.len()
                    ));
                }
                strides
            }
            None => compute_strides(&shape),
        };
        let data_ptr = data.as_ptr();
        Ok(Self {
            data,
            shape: shape.clone(),
            strides: strides.clone(),
            view_cache: UnsafeCell::new(TensorView::new(data_ptr, shape, strides)),
        })
    }

    pub fn from_scalar(value: T) -> Self {
        let data = vec![value];
        let data_ptr = data.as_ptr();
        let shape = Vec::new();
        let strides = Vec::new();
        Self {
            data,
            shape: shape.clone(),
            strides: strides.clone(),
            view_cache: UnsafeCell::new(TensorView::new(data_ptr, shape, strides)),
        }
    }

    pub fn new(data: Vec<T>) -> Self {
        Tensor::from_vec(data)
            .unwrap_or_else(|err| panic!("tensor creation failed: {}", err))
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    pub fn at(&self, indices: &[usize]) -> &T {
        let offset = offset_for(&self.shape, &self.strides, indices)
            .unwrap_or_else(|err| panic!("tensor index error: {}", err));
        &self.data[offset]
    }

    pub fn view(&self, indices: &[usize]) -> TensorView<T> {
        let (offset, shape, strides) =
            view_parts(&self.shape, &self.strides, indices)
                .unwrap_or_else(|err| panic!("tensor view error: {}", err));
        TensorView::new(unsafe { self.data.as_ptr().add(offset) }, shape, strides)
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }
}

impl<T, const N: usize> Index<[usize; N]> for Tensor<T> {
    type Output = TensorView<T>;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let view = self.view(&index);
        unsafe {
            *self.view_cache.get() = view;
            &*self.view_cache.get()
        }
    }
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (idx, dim) in shape.iter().rev().enumerate() {
        let i = shape.len() - 1 - idx;
        strides[i] = stride;
        stride = stride.saturating_mul(*dim);
    }
    strides
}

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let a_dim = dim_from_end(a, i);
        let b_dim = dim_from_end(b, i);
        match (a_dim, b_dim) {
            (Some(x), Some(y)) if x == y => out.push(x),
            (Some(1), Some(y)) => out.push(y),
            (Some(x), Some(1)) => out.push(x),
            (None, Some(y)) => out.push(y),
            (Some(x), None) => out.push(x),
            _ => {
                return Err(anyhow!(
                    "broadcast shape mismatch for {:?} and {:?}",
                    a,
                    b
                ))
            }
        }
    }
    out.reverse();
    Ok(out)
}

pub fn broadcast_to_vec<T: Clone>(tensor: &Tensor<T>, out_shape: &[usize]) -> Result<Vec<T>> {
    if tensor.shape == out_shape {
        return Ok(tensor.data.clone());
    }
    let out_len = numel(out_shape);
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let rank_out = out_shape.len();
    let rank_in = tensor.shape.len();
    let mut aligned_shape = vec![1; rank_out.saturating_sub(rank_in)];
    aligned_shape.extend_from_slice(&tensor.shape);
    let mut aligned_strides = vec![0; rank_out.saturating_sub(rank_in)];
    aligned_strides.extend_from_slice(&tensor.strides);
    let mut out = Vec::with_capacity(out_len);
    for linear in 0..out_len {
        let coords = linear_to_indices(linear, out_shape);
        let mut offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            let in_dim = aligned_shape[dim];
            let stride = if in_dim == 1 { 0 } else { aligned_strides[dim] };
            let in_coord = if in_dim == 1 { 0 } else { *coord };
            offset = offset.saturating_add(in_coord.saturating_mul(stride));
        }
        out.push(tensor.data[offset].clone());
    }
    Ok(out)
}

fn dim_from_end(shape: &[usize], idx_from_end: usize) -> Option<usize> {
    if idx_from_end >= shape.len() {
        return None;
    }
    Some(shape[shape.len() - 1 - idx_from_end])
}

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    strides == compute_strides(shape)
}

fn offset_for(shape: &[usize], strides: &[usize], indices: &[usize]) -> Result<usize> {
    if shape.len() != indices.len() {
        return Err(anyhow!(
            "expected {} indices, got {}",
            shape.len(),
            indices.len()
        ));
    }
    let mut offset = 0usize;
    for ((dim, stride), idx) in shape.iter().zip(strides.iter()).zip(indices.iter()) {
        if *idx >= *dim {
            return Err(anyhow!("index {} out of bounds for dim {}", idx, dim));
        }
        offset = offset.saturating_add(idx.saturating_mul(*stride));
    }
    Ok(offset)
}

fn view_parts(
    shape: &[usize],
    strides: &[usize],
    indices: &[usize],
) -> Result<(usize, Vec<usize>, Vec<usize>)> {
    if indices.len() > shape.len() {
        return Err(anyhow!(
            "too many indices: got {}, shape has {} dims",
            indices.len(),
            shape.len()
        ));
    }
    let mut offset = 0usize;
    for (idx, (dim, stride)) in indices.iter().zip(shape.iter().zip(strides.iter())) {
        if *idx >= *dim {
            return Err(anyhow!("index {} out of bounds for dim {}", idx, dim));
        }
        offset = offset.saturating_add(idx.saturating_mul(*stride));
    }
    Ok((
        offset,
        shape[indices.len()..].to_vec(),
        strides[indices.len()..].to_vec(),
    ))
}

fn linear_to_indices(linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut rem = linear;
    let mut out = Vec::with_capacity(shape.len());
    let strides = compute_strides(shape);
    for (dim, stride) in shape.iter().zip(strides.iter()) {
        if *stride == 0 {
            out.push(0);
        } else {
            let coord = rem / *stride;
            out.push(coord.min(dim.saturating_sub(1)));
            rem %= *stride;
        }
    }
    out
}

pub trait TensorElement: Sized + Clone {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>>;
    fn into_value(tensor: Tensor<Self>) -> TensorValue;
}

impl<T> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        Tensor::new(value)
    }
}

impl TensorElement for f32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F32(tensor)
    }
}

impl TensorElement for f64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F64(tensor)
    }
}

impl TensorElement for i8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I8(tensor)
    }
}

impl TensorElement for i16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I16(tensor)
    }
}

impl TensorElement for i32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I32(tensor)
    }
}

impl TensorElement for i64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I64(tensor)
    }
}

impl TensorElement for u8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U8(tensor)
    }
}

impl TensorElement for u16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U16(tensor)
    }
}

impl TensorElement for u32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U32(tensor)
    }
}

impl TensorElement for u64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U64(tensor)
    }
}

impl TensorElement for bool {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::Bool(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::Bool(tensor)
    }
}

impl TensorElement for F16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F16(tensor)
    }
}

impl TensorElement for Bitset {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::Bitset(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::Bitset(tensor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    I8,
    I16,
    F32,
    F64,
    U8,
    U16,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Bitset,
    F16,
}

impl DType {
    pub fn from_ident(ident: &str) -> Result<Self> {
        match ident {
            "i8" => Ok(DType::I8),
            "i16" => Ok(DType::I16),
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            "u8" => Ok(DType::U8),
            "u16" => Ok(DType::U16),
            "i32" => Ok(DType::I32),
            "i64" => Ok(DType::I64),
            "u32" => Ok(DType::U32),
            "u64" => Ok(DType::U64),
            "bool" => Ok(DType::Bool),
            "bitset" => Ok(DType::Bitset),
            "f16" => Ok(DType::F16),
            _ => Err(anyhow!("unsupported dtype: {}", ident)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorValue {
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    U32(Tensor<u32>),
    U64(Tensor<u64>),
    Bool(Tensor<bool>),
    Bitset(Tensor<Bitset>),
    F16(Tensor<F16>),
}

// TensorValue is moved across threads but not shared concurrently.
unsafe impl Send for TensorValue {}

impl TensorValue {
    pub fn dtype(&self) -> DType {
        match self {
            TensorValue::I8(_) => DType::I8,
            TensorValue::I16(_) => DType::I16,
            TensorValue::F32(_) => DType::F32,
            TensorValue::F64(_) => DType::F64,
            TensorValue::U8(_) => DType::U8,
            TensorValue::U16(_) => DType::U16,
            TensorValue::I32(_) => DType::I32,
            TensorValue::I64(_) => DType::I64,
            TensorValue::U32(_) => DType::U32,
            TensorValue::U64(_) => DType::U64,
            TensorValue::Bool(_) => DType::Bool,
            TensorValue::Bitset(_) => DType::Bitset,
            TensorValue::F16(_) => DType::F16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorValue::I8(tensor) => tensor.len(),
            TensorValue::I16(tensor) => tensor.len(),
            TensorValue::F32(tensor) => tensor.len(),
            TensorValue::F64(tensor) => tensor.len(),
            TensorValue::U8(tensor) => tensor.len(),
            TensorValue::U16(tensor) => tensor.len(),
            TensorValue::I32(tensor) => tensor.len(),
            TensorValue::I64(tensor) => tensor.len(),
            TensorValue::U32(tensor) => tensor.len(),
            TensorValue::U64(tensor) => tensor.len(),
            TensorValue::Bool(tensor) => tensor.len(),
            TensorValue::Bitset(tensor) => tensor.len(),
            TensorValue::F16(tensor) => tensor.len(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            TensorValue::I8(tensor) => tensor.shape(),
            TensorValue::I16(tensor) => tensor.shape(),
            TensorValue::F32(tensor) => tensor.shape(),
            TensorValue::F64(tensor) => tensor.shape(),
            TensorValue::U8(tensor) => tensor.shape(),
            TensorValue::U16(tensor) => tensor.shape(),
            TensorValue::I32(tensor) => tensor.shape(),
            TensorValue::I64(tensor) => tensor.shape(),
            TensorValue::U32(tensor) => tensor.shape(),
            TensorValue::U64(tensor) => tensor.shape(),
            TensorValue::Bool(tensor) => tensor.shape(),
            TensorValue::Bitset(tensor) => tensor.shape(),
            TensorValue::F16(tensor) => tensor.shape(),
        }
    }

    pub fn strides(&self) -> &[usize] {
        match self {
            TensorValue::I8(tensor) => tensor.strides(),
            TensorValue::I16(tensor) => tensor.strides(),
            TensorValue::F32(tensor) => tensor.strides(),
            TensorValue::F64(tensor) => tensor.strides(),
            TensorValue::U8(tensor) => tensor.strides(),
            TensorValue::U16(tensor) => tensor.strides(),
            TensorValue::I32(tensor) => tensor.strides(),
            TensorValue::I64(tensor) => tensor.strides(),
            TensorValue::U32(tensor) => tensor.strides(),
            TensorValue::U64(tensor) => tensor.strides(),
            TensorValue::Bool(tensor) => tensor.strides(),
            TensorValue::Bitset(tensor) => tensor.strides(),
            TensorValue::F16(tensor) => tensor.strides(),
        }
    }

    pub fn zeros(dtype: DType, shape: &[usize]) -> Self {
        let len = numel(shape);
        match dtype {
            DType::I8 => TensorValue::I8(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I16 => TensorValue::I16(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F32 => TensorValue::F32(
                Tensor::from_vec_with_opts(vec![0.0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F64 => TensorValue::F64(
                Tensor::from_vec_with_opts(vec![0.0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U8 => TensorValue::U8(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U16 => TensorValue::U16(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I32 => TensorValue::I32(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I64 => TensorValue::I64(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U32 => TensorValue::U32(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U64 => TensorValue::U64(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::Bool => TensorValue::Bool(
                Tensor::from_vec_with_opts(vec![false; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::Bitset => TensorValue::Bitset(
                Tensor::from_vec_with_opts(vec![Bitset { bits: 0 }; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F16 => TensorValue::F16(
                Tensor::from_vec_with_opts(vec![F16 { bits: 0 }; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
        }
    }

    pub fn as_i8(&self) -> Result<&Tensor<i8>> {
        match self {
            TensorValue::I8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i8 tensor")),
        }
    }

    pub fn as_i16(&self) -> Result<&Tensor<i16>> {
        match self {
            TensorValue::I16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i16 tensor")),
        }
    }

    pub fn as_f32(&self) -> Result<&Tensor<f32>> {
        match self {
            TensorValue::F32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f32 tensor")),
        }
    }

    pub fn as_f64(&self) -> Result<&Tensor<f64>> {
        match self {
            TensorValue::F64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f64 tensor")),
        }
    }

    pub fn as_u8(&self) -> Result<&Tensor<u8>> {
        match self {
            TensorValue::U8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u8 tensor")),
        }
    }

    pub fn as_u16(&self) -> Result<&Tensor<u16>> {
        match self {
            TensorValue::U16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u16 tensor")),
        }
    }

    pub fn as_i32(&self) -> Result<&Tensor<i32>> {
        match self {
            TensorValue::I32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i32 tensor")),
        }
    }

    pub fn as_i64(&self) -> Result<&Tensor<i64>> {
        match self {
            TensorValue::I64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i64 tensor")),
        }
    }

    pub fn as_u32(&self) -> Result<&Tensor<u32>> {
        match self {
            TensorValue::U32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u32 tensor")),
        }
    }

    pub fn as_u64(&self) -> Result<&Tensor<u64>> {
        match self {
            TensorValue::U64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u64 tensor")),
        }
    }

    pub fn as_bool(&self) -> Result<&Tensor<bool>> {
        match self {
            TensorValue::Bool(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bool tensor")),
        }
    }

    pub fn as_bitset(&self) -> Result<&Tensor<Bitset>> {
        match self {
            TensorValue::Bitset(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bitset tensor")),
        }
    }

    pub fn as_f16(&self) -> Result<&Tensor<F16>> {
        match self {
            TensorValue::F16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f16 tensor")),
        }
    }
}

impl From<Tensor<i8>> for TensorValue {
    fn from(value: Tensor<i8>) -> Self {
        TensorValue::I8(value)
    }
}

impl From<Tensor<i16>> for TensorValue {
    fn from(value: Tensor<i16>) -> Self {
        TensorValue::I16(value)
    }
}

impl From<Tensor<f32>> for TensorValue {
    fn from(value: Tensor<f32>) -> Self {
        TensorValue::F32(value)
    }
}

impl From<Tensor<f64>> for TensorValue {
    fn from(value: Tensor<f64>) -> Self {
        TensorValue::F64(value)
    }
}

impl From<Tensor<i32>> for TensorValue {
    fn from(value: Tensor<i32>) -> Self {
        TensorValue::I32(value)
    }
}

impl From<Tensor<i64>> for TensorValue {
    fn from(value: Tensor<i64>) -> Self {
        TensorValue::I64(value)
    }
}

impl From<Tensor<u8>> for TensorValue {
    fn from(value: Tensor<u8>) -> Self {
        TensorValue::U8(value)
    }
}

impl From<Tensor<u16>> for TensorValue {
    fn from(value: Tensor<u16>) -> Self {
        TensorValue::U16(value)
    }
}

impl From<Tensor<u32>> for TensorValue {
    fn from(value: Tensor<u32>) -> Self {
        TensorValue::U32(value)
    }
}

impl From<Tensor<u64>> for TensorValue {
    fn from(value: Tensor<u64>) -> Self {
        TensorValue::U64(value)
    }
}

impl From<Tensor<bool>> for TensorValue {
    fn from(value: Tensor<bool>) -> Self {
        TensorValue::Bool(value)
    }
}

impl From<Tensor<Bitset>> for TensorValue {
    fn from(value: Tensor<Bitset>) -> Self {
        TensorValue::Bitset(value)
    }
}

impl From<Tensor<F16>> for TensorValue {
    fn from(value: Tensor<F16>) -> Self {
        TensorValue::F16(value)
    }
}

impl From<i8> for TensorValue {
    fn from(value: i8) -> Self {
        TensorValue::I8(Tensor::from_scalar(value))
    }
}

impl From<i16> for TensorValue {
    fn from(value: i16) -> Self {
        TensorValue::I16(Tensor::from_scalar(value))
    }
}

impl From<i32> for TensorValue {
    fn from(value: i32) -> Self {
        TensorValue::I32(Tensor::from_scalar(value))
    }
}

impl From<i64> for TensorValue {
    fn from(value: i64) -> Self {
        TensorValue::I64(Tensor::from_scalar(value))
    }
}

impl From<u8> for TensorValue {
    fn from(value: u8) -> Self {
        TensorValue::U8(Tensor::from_scalar(value))
    }
}

impl From<u16> for TensorValue {
    fn from(value: u16) -> Self {
        TensorValue::U16(Tensor::from_scalar(value))
    }
}

impl From<u32> for TensorValue {
    fn from(value: u32) -> Self {
        TensorValue::U32(Tensor::from_scalar(value))
    }
}

impl From<u64> for TensorValue {
    fn from(value: u64) -> Self {
        TensorValue::U64(Tensor::from_scalar(value))
    }
}

impl From<f32> for TensorValue {
    fn from(value: f32) -> Self {
        TensorValue::F32(Tensor::from_scalar(value))
    }
}

impl From<f64> for TensorValue {
    fn from(value: f64) -> Self {
        TensorValue::F64(Tensor::from_scalar(value))
    }
}

impl From<bool> for TensorValue {
    fn from(value: bool) -> Self {
        TensorValue::Bool(Tensor::from_scalar(value))
    }
}

impl From<Bitset> for TensorValue {
    fn from(value: Bitset) -> Self {
        TensorValue::Bitset(Tensor::from_scalar(value))
    }
}

impl From<F16> for TensorValue {
    fn from(value: F16) -> Self {
        TensorValue::F16(Tensor::from_scalar(value))
    }
}
