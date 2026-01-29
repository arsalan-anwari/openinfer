use anyhow::{anyhow, Result};
use std::cell::UnsafeCell;
use std::ops::Index;

use super::shape::{is_contiguous, linear_to_indices, numel, offset_for, view_parts, compute_strides};

#[derive(Debug, Clone, Default)]
pub struct TensorOptions {
    pub shape: Option<Vec<usize>>,
    pub strides: Option<Vec<usize>>,
    pub allow_len_mismatch: bool,
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
        if !opts.allow_len_mismatch && expected != data.len() {
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
