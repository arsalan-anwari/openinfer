use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::tensor::{numel, BF16, F8E5M2, I1, I2, I4, Tensor, TensorOptions};

pub struct Random<T> {
    rng: StdRng,
    _marker: PhantomData<T>,
}

impl<T> Random<T>
where
    T: RandomValue,
{
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            _marker: PhantomData,
        }
    }

    pub fn generate(range: (T, T), len: usize) -> Result<Tensor<T>> {
        Self::generate_with_seed_opts(0, range, len, TensorOptions::default())
    }

    pub fn generate_with_opts(range: (T, T), len: usize, opts: TensorOptions) -> Result<Tensor<T>> {
        Self::generate_with_seed_opts(0, range, len, opts)
    }

    pub fn generate_with_seed(seed: u64, range: (T, T), len: usize) -> Result<Tensor<T>> {
        Self::generate_with_seed_opts(seed, range, len, TensorOptions::default())
    }

    pub fn generate_with_seed_opts(
        seed: u64,
        range: (T, T),
        len: usize,
        opts: TensorOptions,
    ) -> Result<Tensor<T>> {
        let mut rng = StdRng::seed_from_u64(seed);
        generate_with_rng::<T>(&mut rng, range, len, opts)
    }

    pub fn next(&mut self, range: (T, T), len: usize) -> Result<Tensor<T>> {
        self.next_with_opts(range, len, TensorOptions::default())
    }

    pub fn next_with_opts(
        &mut self,
        range: (T, T),
        len: usize,
        opts: TensorOptions,
    ) -> Result<Tensor<T>> {
        generate_with_rng::<T>(&mut self.rng, range, len, opts)
    }
}

fn generate_with_rng<T: RandomValue>(
    rng: &mut StdRng,
    range: (T, T),
    len: usize,
    opts: TensorOptions,
) -> Result<Tensor<T>> {
    if opts.shape.is_none() && opts.strides.is_some() {
        return Err(anyhow!("random tensor strides require an explicit shape"));
    }
    let shape = match opts.shape.as_ref() {
        Some(shape) => {
            let expected = numel(shape);
            if expected != len {
                return Err(anyhow!(
                    "random tensor shape {:?} expects {} values, got {}",
                    shape,
                    expected,
                    len
                ));
            }
            shape.clone()
        }
        None => vec![len],
    };
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(T::sample(rng, range)?);
    }
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape),
            strides: opts.strides,
            allow_len_mismatch: opts.allow_len_mismatch,
        },
    )
}

pub trait RandomValue: Sized + Copy {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self>;
}

impl RandomValue for f32 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for f64 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for i8 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for i16 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for i32 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for i64 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for u8 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for u16 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for u32 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for u64 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        Ok(rng.gen_range(range.0..=range.1))
    }
}

impl RandomValue for crate::tensor::F16 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        let value = rng.gen_range(range.0.to_f32()..=range.1.to_f32());
        Ok(crate::tensor::F16::from_f32(value))
    }
}

impl RandomValue for BF16 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        let value = rng.gen_range(range.0.to_f32()..=range.1.to_f32());
        Ok(BF16::from_f32(value))
    }
}

impl RandomValue for F8E5M2 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        let value = rng.gen_range(range.0.to_f32()..=range.1.to_f32());
        Ok(F8E5M2::from_f32(value))
    }
}

impl RandomValue for I4 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        let value = rng.gen_range(range.0.to_i8()..=range.1.to_i8());
        Ok(I4::from_i8(value))
    }
}

impl RandomValue for I2 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        let value = rng.gen_range(range.0.to_i8()..=range.1.to_i8());
        Ok(I2::from_i8(value))
    }
}

impl RandomValue for I1 {
    fn sample(rng: &mut StdRng, range: (Self, Self)) -> Result<Self> {
        let value = rng.gen_range(range.0.to_i8()..=range.1.to_i8());
        Ok(I1::from_i8(value))
    }
}
