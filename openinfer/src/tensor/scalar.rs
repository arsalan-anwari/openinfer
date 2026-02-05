use serde::{Deserialize, Serialize};

/// Packed bitset storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bitset {
    pub bits: u8,
}

/// Brain-float16 (BF16) scalar storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BF16 {
    pub bits: u16,
}

impl BF16 {
    /// Convert from f32 to BF16.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let rounding = 0x7fff + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding);
        Self {
            bits: (rounded >> 16) as u16,
        }
    }

    /// Convert BF16 to f32.
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.bits as u32) << 16)
    }
}

/// IEEE 754 half-precision (F16) scalar storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct F16 {
    pub bits: u16,
}

impl F16 {
    /// Convert from f32 to F16.
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

    /// Convert F16 to f32.
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

/// 8-bit float storage (custom format).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct F8 {
    pub bits: u8,
}

impl F8 {
    /// Convert from f32 to F8.
    pub fn from_f32(value: f32) -> Self {
        if value.is_nan() {
            return Self { bits: 0x7d };
        }
        if value.is_infinite() {
            return Self {
                bits: ((value.is_sign_negative() as u8) << 7) | 0x7c,
            };
        }
        if value == 0.0 {
            return Self {
                bits: ((value.is_sign_negative() as u8) << 7),
            };
        }

        let bits = value.to_bits();
        let sign = ((bits >> 31) & 1) as u8;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7fffff;

        if exp == 0 {
            return Self { bits: sign << 7 };
        }

        let exp_unbiased = exp - 127;
        let mut exp8 = exp_unbiased + 15;
        let mantissa = mant | 0x800000;

        if exp8 >= 31 {
            return Self { bits: (sign << 7) | 0x7c };
        }

        if exp8 <= 0 {
            let shift = (1 - exp8) as u32;
            let mant_shift = 21 + shift;
            if mant_shift >= 32 {
                return Self { bits: sign << 7 };
            }
            let rounding_bit = 1u32 << (mant_shift - 1);
            let rounded = mantissa.wrapping_add(rounding_bit);
            let mant2 = (rounded >> mant_shift) as u8 & 0x03;
            return Self {
                bits: (sign << 7) | mant2,
            };
        }

        let rounding_bit = 1u32 << 20;
        let rounded = mantissa.wrapping_add(rounding_bit);
        let mant2 = (rounded >> 21) as u8 & 0x03;
        if mant2 == 0x04 {
            exp8 += 1;
            if exp8 >= 31 {
                return Self { bits: (sign << 7) | 0x7c };
            }
        }
        Self {
            bits: (sign << 7) | ((exp8 as u8) << 2) | (mant2 & 0x03),
        }
    }

    /// Convert F8 to f32.
    pub fn to_f32(self) -> f32 {
        let sign = ((self.bits >> 7) & 1) as u32;
        let exp = ((self.bits >> 2) & 0x1f) as i32;
        let mant = (self.bits & 0x03) as u32;
        let sign_bits = sign << 31;

        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign_bits);
            }
            let frac = (mant as f32) / 4.0;
            let value = (1.0 / (1u32 << 14) as f32) * frac;
            return if sign == 1 { -value } else { value };
        }
        if exp == 31 {
            let bits = sign_bits | 0x7f800000 | (mant << 21);
            return f32::from_bits(bits);
        }
        let exp32 = (exp - 15 + 127) as u32;
        let mant32 = mant << 21;
        f32::from_bits(sign_bits | (exp32 << 23) | mant32)
    }
}

/// 4-bit signed integer storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct I4 {
    pub bits: u8,
}

/// 2-bit signed integer storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct I2 {
    pub bits: u8,
}

/// 1-bit signed integer storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct I1 {
    pub bits: u8,
}

/// 4-bit unsigned integer storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct U4 {
    pub bits: u8,
}

/// 2-bit unsigned integer storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct U2 {
    pub bits: u8,
}

/// 1-bit unsigned integer storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct U1 {
    pub bits: u8,
}

/// 2-bit ternary storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct T2 {
    pub bits: u8,
}

/// 1-bit ternary storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct T1 {
    pub bits: u8,
}

fn sign_extend(bits: u8, width: u8) -> i8 {
    let shift = 8 - width;
    ((bits << shift) as i8) >> shift
}

impl I4 {
    /// Create an I4 value from i8.
    pub fn from_i8(value: i8) -> Self {
        Self {
            bits: (value as u8) & 0x0f,
        }
    }

    /// Convert I4 to i8.
    pub fn to_i8(self) -> i8 {
        sign_extend(self.bits & 0x0f, 4)
    }
}

impl I2 {
    /// Create an I2 value from i8.
    pub fn from_i8(value: i8) -> Self {
        Self {
            bits: (value as u8) & 0x03,
        }
    }

    /// Convert I2 to i8.
    pub fn to_i8(self) -> i8 {
        sign_extend(self.bits & 0x03, 2)
    }
}

impl I1 {
    /// Create an I1 value from i8.
    pub fn from_i8(value: i8) -> Self {
        Self { bits: (value as u8) & 0x01 }
    }

    /// Convert I1 to i8.
    pub fn to_i8(self) -> i8 {
        sign_extend(self.bits & 0x01, 1)
    }
}

impl U4 {
    /// Create a U4 value from u8.
    pub fn from_u8(value: u8) -> Self {
        Self { bits: value & 0x0f }
    }

    /// Convert U4 to u8.
    pub fn to_u8(self) -> u8 {
        self.bits & 0x0f
    }
}

impl U2 {
    /// Create a U2 value from u8.
    pub fn from_u8(value: u8) -> Self {
        Self { bits: value & 0x03 }
    }

    /// Convert U2 to u8.
    pub fn to_u8(self) -> u8 {
        self.bits & 0x03
    }
}

impl U1 {
    /// Create a U1 value from u8.
    pub fn from_u8(value: u8) -> Self {
        Self { bits: value & 0x01 }
    }

    /// Convert U1 to u8.
    pub fn to_u8(self) -> u8 {
        self.bits & 0x01
    }
}

impl T2 {
    /// Create a T2 value from i8.
    pub fn from_i8(value: i8) -> Self {
        Self {
            bits: (value as u8) & 0x03,
        }
    }

    /// Convert T2 to i8.
    pub fn to_i8(self) -> i8 {
        sign_extend(self.bits & 0x03, 2)
    }
}

impl T1 {
    /// Create a T1 value from i8.
    pub fn from_i8(value: i8) -> Self {
        let bits = if value < 0 { 0 } else { 1 };
        Self { bits }
    }

    /// Convert T1 to i8.
    pub fn to_i8(self) -> i8 {
        if (self.bits & 0x01) == 0 {
            -1
        } else {
            1
        }
    }
}
