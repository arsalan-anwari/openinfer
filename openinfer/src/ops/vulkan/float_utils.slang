#include "packed_utils.slang"

uint load_u16(ByteAddressBuffer buf, uint index) {
    uint byte_index = index * 2u;
    uint low = load_u8(buf, byte_index);
    uint high = load_u8(buf, byte_index + 1u);
    return low | (high << 8u);
}

uint load_u16_rw(RWByteAddressBuffer buf, uint index) {
    uint byte_index = index * 2u;
    uint low = load_u8_rw(buf, byte_index);
    uint high = load_u8_rw(buf, byte_index + 1u);
    return low | (high << 8u);
}

void store_u16(RWByteAddressBuffer buf, uint index, uint value) {
    uint byte_index = index * 2u;
    store_u8(buf, byte_index, value & 0xFFu);
    store_u8(buf, byte_index + 1u, (value >> 8u) & 0xFFu);
}

float bf16_to_f32(uint bits) {
    return asfloat(bits << 16u);
}

uint f32_to_bf16_bits(float value) {
    uint bits = asuint(value);
    uint rounding = 0x7fffu + ((bits >> 16u) & 1u);
    uint rounded = bits + rounding;
    return (rounded >> 16u) & 0xFFFFu;
}

float f16_to_f32(uint bits16) {
    uint sign = (bits16 & 0x8000u) << 16u;
    uint exp = (bits16 >> 10u) & 0x1fu;
    uint mant = bits16 & 0x03ffu;
    uint bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            uint mant32 = mant;
            int exp32 = -1;
            while ((mant32 & 0x0400u) == 0u) {
                mant32 <<= 1u;
                exp32 -= 1;
            }
            mant32 &= 0x03ffu;
            uint exp32u = (uint)(exp32 + 1 + 127 - 15);
            bits = sign | (exp32u << 23u) | (mant32 << 13u);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        uint exp32u = exp + (127u - 15u);
        bits = sign | (exp32u << 23u) | (mant << 13u);
    }
    return asfloat(bits);
}

uint f32_to_f16_bits(float value) {
    uint bits = asuint(value);
    uint sign = (bits >> 16u) & 0x8000u;
    uint exp = (bits >> 23u) & 0xffu;
    uint mant = bits & 0x7fffffu;
    if (exp == 0u) {
        return sign;
    }
    if (exp == 0xffu) {
        if (mant == 0u) {
            return sign | 0x7c00u;
        }
        return sign | 0x7c00u | ((mant >> 13u) & 0x03ffu) | 1u;
    }
    int exp16 = int(exp) - 127 + 15;
    if (exp16 >= 0x1f) {
        return sign | 0x7c00u;
    }
    if (exp16 <= 0) {
        if (exp16 < -10) {
            return sign;
        }
        uint mant16 = mant | 0x800000u;
        uint shift = uint(14 - exp16);
        uint half = mant16 >> shift;
        if (((mant16 >> (shift - 1u)) & 1u) != 0u) {
            half += 1u;
        }
        return sign | (half & 0x03ffu);
    }
    uint half = (uint(exp16) << 10u) | ((mant >> 13u) & 0x03ffu);
    if (((mant >> 12u) & 1u) != 0u) {
        half += 1u;
    }
    return sign | (half & 0x7fffu);
}

float f8_to_f32(uint bits8) {
    uint sign = (bits8 >> 7u) & 1u;
    uint exp = (bits8 >> 2u) & 0x1fu;
    uint mant = bits8 & 0x03u;
    uint sign_bits = sign << 31u;
    if (exp == 0u) {
        if (mant == 0u) {
            return asfloat(sign_bits);
        }
        float frac = (float)mant / 4.0f;
        float value = (1.0f / (float)(1u << 14u)) * frac;
        return sign == 1u ? -value : value;
    }
    if (exp == 31u) {
        uint bits = sign_bits | 0x7f800000u | (mant << 21u);
        return asfloat(bits);
    }
    uint exp32 = exp - 15u + 127u;
    uint mant32 = mant << 21u;
    return asfloat(sign_bits | (exp32 << 23u) | mant32);
}

uint f32_to_f8_bits(float value) {
    uint bits = asuint(value);
    uint sign = (bits >> 31u) & 1u;
    int exp = int((bits >> 23u) & 0xffu);
    uint mant = bits & 0x7fffffu;
    if (exp == 255) {
        if (mant != 0u) {
            return 0x7du;
        }
        return (sign << 7u) | 0x7cu;
    }
    if (exp == 0 && mant == 0u) {
        return sign << 7u;
    }
    if (exp == 0) {
        return sign << 7u;
    }
    int exp_unbiased = exp - 127;
    int exp8 = exp_unbiased + 15;
    uint mantissa = mant | 0x800000u;
    if (exp8 >= 31) {
        return (sign << 7u) | 0x7cu;
    }
    if (exp8 <= 0) {
        uint shift = uint(1 - exp8);
        uint mant_shift = 21u + shift;
        if (mant_shift >= 32u) {
            return sign << 7u;
        }
        uint rounding_bit = 1u << (mant_shift - 1u);
        uint rounded = mantissa + rounding_bit;
        uint mant2 = (rounded >> mant_shift) & 0x03u;
        return (sign << 7u) | (mant2 & 0x03u);
    }
    uint rounding_bit = 1u << 20u;
    uint rounded = mantissa + rounding_bit;
    uint mant2 = (rounded >> 21u) & 0x03u;
    if (mant2 == 0x04u) {
        exp8 += 1;
        if (exp8 >= 31) {
            return (sign << 7u) | 0x7cu;
        }
    }
    return (sign << 7u) | (uint(exp8) << 2u) | (mant2 & 0x03u);
}

float read_f16(ByteAddressBuffer buf, uint index) {
    return f16_to_f32(load_u16(buf, index));
}

float read_bf16(ByteAddressBuffer buf, uint index) {
    return bf16_to_f32(load_u16(buf, index));
}

float read_f8(ByteAddressBuffer buf, uint index) {
    return f8_to_f32(load_u8(buf, index));
}

void write_f16(RWByteAddressBuffer buf, uint index, float value) {
    store_u16(buf, index, f32_to_f16_bits(value));
}

void write_bf16(RWByteAddressBuffer buf, uint index, float value) {
    store_u16(buf, index, f32_to_bf16_bits(value));
}

void write_f8(RWByteAddressBuffer buf, uint index, float value) {
    store_u8(buf, index, f32_to_f8_bits(value));
}
