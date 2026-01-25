#[cfg(any(feature = "avx", feature = "avx2"))]

use anyhow::Result;

#[cfg(any(feature = "avx", feature = "avx2"))]

pub enum OutputBuf<'a, T>
where
    T: Default + Copy,
{
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

#[cfg(any(feature = "avx", feature = "avx2"))]

impl<'a, T> OutputBuf<'a, T>
where
    T: Default + Copy,
{
    pub fn new(len: usize, output: Option<&'a mut [T]>, err: &'static str) -> Result<Self> {
        if let Some(out) = output {
            if out.len() != len {
                return Err(anyhow::anyhow!(err));
            }
            Ok(Self::Borrowed(out))
        } else {
            Ok(Self::Owned(vec![T::default(); len]))
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(vec) => vec.as_mut_slice(),
        }
    }

    pub fn into_result(self) -> Option<Vec<T>> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(vec) => Some(vec),
        }
    }
}
