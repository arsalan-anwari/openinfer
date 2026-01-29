use crate::tensor::{Bitset, BF16, F16, F8, I1, I2, I4, U1, U2, U4};

pub trait MulElement: Copy {
    fn mul(self, rhs: Self) -> Self;
}

pub trait SignedInput: Copy {
    fn to_i64(self) -> i64;
}

pub trait UnsignedInput: Copy {
    fn to_u64(self) -> u64;
}

pub trait SignedAcc: Copy {
    fn from_i64(value: i64) -> Self;
}

pub trait UnsignedAcc: Copy {
    fn from_u64(value: u64) -> Self;
}

impl MulElement for f32 {
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MulElement for f64 {
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MulElement for i8 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for i16 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for i32 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for i64 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u8 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u16 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u32 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u64 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for bool {
    fn mul(self, rhs: Self) -> Self {
        self & rhs
    }
}

impl MulElement for Bitset {
    fn mul(self, rhs: Self) -> Self {
        Bitset {
            bits: self.bits.wrapping_mul(rhs.bits),
        }
    }
}

impl MulElement for F16 {
    fn mul(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulElement for BF16 {
    fn mul(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulElement for F8 {
    fn mul(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl SignedInput for i8 {
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SignedInput for I1 {
    fn to_i64(self) -> i64 {
        self.to_i8() as i64
    }
}

impl SignedInput for I2 {
    fn to_i64(self) -> i64 {
        self.to_i8() as i64
    }
}

impl SignedInput for I4 {
    fn to_i64(self) -> i64 {
        self.to_i8() as i64
    }
}

impl SignedInput for i16 {
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SignedInput for i32 {
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SignedInput for i64 {
    fn to_i64(self) -> i64 {
        self
    }
}

impl UnsignedInput for U1 {
    fn to_u64(self) -> u64 {
        self.to_u8() as u64
    }
}

impl UnsignedInput for U2 {
    fn to_u64(self) -> u64 {
        self.to_u8() as u64
    }
}

impl UnsignedInput for U4 {
    fn to_u64(self) -> u64 {
        self.to_u8() as u64
    }
}

impl UnsignedInput for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl UnsignedInput for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl UnsignedInput for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl UnsignedInput for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

impl SignedAcc for i8 {
    fn from_i64(value: i64) -> Self {
        value as i8
    }
}

impl SignedAcc for i16 {
    fn from_i64(value: i64) -> Self {
        value as i16
    }
}

impl SignedAcc for i32 {
    fn from_i64(value: i64) -> Self {
        value as i32
    }
}

impl SignedAcc for i64 {
    fn from_i64(value: i64) -> Self {
        value
    }
}

impl UnsignedAcc for u8 {
    fn from_u64(value: u64) -> Self {
        value as u8
    }
}

impl UnsignedAcc for u16 {
    fn from_u64(value: u64) -> Self {
        value as u16
    }
}

impl UnsignedAcc for u32 {
    fn from_u64(value: u64) -> Self {
        value as u32
    }
}

impl UnsignedAcc for u64 {
    fn from_u64(value: u64) -> Self {
        value
    }
}
