use crate::tensor::{BF16, F16, F8};

pub trait SubElement: Copy {
    fn sub(self, rhs: Self) -> Self;
}

impl SubElement for f32 {
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl SubElement for f64 {
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl SubElement for i8 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for i16 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for i32 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for i64 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for u8 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for u16 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for u32 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for u64 {
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl SubElement for F16 {
    fn sub(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubElement for BF16 {
    fn sub(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubElement for F8 {
    fn sub(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() - rhs.to_f32())
    }
}
