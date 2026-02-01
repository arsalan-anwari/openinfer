use crate::tensor::{Bitset, BF16, F16, F8, I1, I2, I4, U1, U2, U4};

pub trait MatmulElement: Copy {
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
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

impl MatmulElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MatmulElement for f64 {
    fn zero() -> Self {
        0.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MatmulElement for i8 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for i16 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for i32 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for i64 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u8 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u16 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u32 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u64 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for bool {
    fn zero() -> Self {
        false
    }

    fn add(self, rhs: Self) -> Self {
        let sum = (self as u8).wrapping_add(rhs as u8);
        sum != 0
    }

    fn mul(self, rhs: Self) -> Self {
        self & rhs
    }
}

impl MatmulElement for Bitset {
    fn zero() -> Self {
        Bitset { bits: 0 }
    }

    fn add(self, rhs: Self) -> Self {
        Bitset {
            bits: self.bits.wrapping_add(rhs.bits),
        }
    }

    fn mul(self, rhs: Self) -> Self {
        Bitset {
            bits: self.bits.wrapping_mul(rhs.bits),
        }
    }
}

impl MatmulElement for F16 {
    fn zero() -> Self {
        F16::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() + rhs.to_f32())
    }

    fn mul(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MatmulElement for BF16 {
    fn zero() -> Self {
        BF16::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() + rhs.to_f32())
    }

    fn mul(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MatmulElement for F8 {
    fn zero() -> Self {
        F8::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() + rhs.to_f32())
    }

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
