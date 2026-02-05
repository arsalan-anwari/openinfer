"""Numeric conversion helpers for packed and custom dtypes."""
from __future__ import annotations

import numpy as np


def float_to_bf16_bits(value: float) -> int:
    """Convert a float to BF16 bit representation."""
    bits = np.float32(value).view(np.uint32)
    rounding = np.uint32(0x7FFF + ((bits >> 16) & 1))
    rounded = bits + rounding
    return int(rounded >> 16)


def float_to_f8_bits(value: float) -> int:
    """Convert a float to F8 bit representation."""
    value = float(value)
    if np.isnan(value):
        return 0x7D
    if np.isinf(value):
        return 0xFC if value < 0 else 0x7C
    if value == 0.0:
        return 0x80 if np.signbit(value) else 0x00
    bits = np.float32(value).view(np.uint32)
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0:
        return int(sign << 7)
    exp_unbiased = int(exp) - 127
    exp8 = exp_unbiased + 15
    mantissa = mant | 0x800000
    if exp8 >= 31:
        return int((sign << 7) | 0x7C)
    if exp8 <= 0:
        shift = 1 - exp8
        mant_shift = 21 + shift
        if mant_shift >= 32:
            return int(sign << 7)
        rounded = mantissa + (1 << (mant_shift - 1))
        mant2 = (rounded >> mant_shift) & 0x03
        return int((sign << 7) | mant2)
    rounded = mantissa + (1 << 20)
    mant2 = (rounded >> 21) & 0x03
    if mant2 == 0x04:
        exp8 += 1
        if exp8 >= 31:
            return int((sign << 7) | 0x7C)
        mant2 = 0
    return int((sign << 7) | ((exp8 & 0x1F) << 2) | (mant2 & 0x03))


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Convert BF16 bit arrays to float32."""
    bits32 = bits.astype(np.uint32) << 16
    return bits32.view(np.float32)


def f8_to_f32_scalar(bits: int) -> float:
    """Convert a single F8 byte to float32."""
    sign = (bits >> 7) & 1
    exp = (bits >> 2) & 0x1F
    mant = bits & 0x03
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        frac = mant / 4.0
        value = (2 ** -14) * frac
        return -value if sign else value
    if exp == 31:
        return float("-inf") if sign else float("inf")
    exp32 = exp - 15
    value = (1.0 + mant / 4.0) * (2 ** exp32)
    return -value if sign else value


def f8_to_f32(bits: np.ndarray) -> np.ndarray:
    """Convert an array of F8 bytes to float32."""
    vec = np.vectorize(f8_to_f32_scalar, otypes=[np.float32])
    return vec(bits.astype(np.uint8))
