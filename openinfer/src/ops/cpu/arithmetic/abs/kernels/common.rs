use crate::tensor::{BF16, F16, F8, I1, I2, I4};

pub trait AbsElement: Copy {
    fn abs_value(self) -> Self;
}

pub trait SignedInput: Copy {
    fn to_i64(self) -> i64;
}

pub trait SignedAcc: Copy {
    fn from_i64(value: i64) -> Self;
}


impl AbsElement for f32 {
    fn abs_value(self) -> Self {
        self.abs()
    }
}

impl AbsElement for f64 {
    fn abs_value(self) -> Self {
        self.abs()
    }
}

impl AbsElement for i8 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for i16 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for i32 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for i64 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for F16 {
    fn abs_value(self) -> Self {
        F16::from_f32(self.to_f32().abs())
    }
}

impl AbsElement for BF16 {
    fn abs_value(self) -> Self {
        BF16::from_f32(self.to_f32().abs())
    }
}

impl AbsElement for F8 {
    fn abs_value(self) -> Self {
        F8::from_f32(self.to_f32().abs())
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
