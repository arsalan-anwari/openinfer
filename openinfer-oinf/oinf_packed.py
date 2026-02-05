"""Packing utilities for bit-level tensor formats."""
from __future__ import annotations

import numpy as np

from oinf_common import OinfError


def pack_signed_bits(values: np.ndarray, bits_per: int) -> bytes:
    """Pack signed integers into a compact bitstream."""
    if bits_per <= 0 or bits_per > 8:
        raise OinfError(f"Invalid packed bit width {bits_per}")
    total = int(values.size)
    out = bytearray((total * bits_per + 7) // 8)
    mask = (1 << bits_per) - 1
    for idx, value in enumerate(values.flatten()):
        raw = int(value) & mask
        bit_index = idx * bits_per
        byte_index = bit_index // 8
        shift = bit_index % 8
        out[byte_index] |= (raw << shift) & 0xFF
        if shift + bits_per > 8:
            out[byte_index + 1] |= (raw >> (8 - shift)) & 0xFF
    return bytes(out)


def pack_unsigned_bits(values: np.ndarray, bits_per: int) -> bytes:
    """Pack unsigned integers into a compact bitstream."""
    if bits_per <= 0 or bits_per > 8:
        raise OinfError(f"Invalid packed bit width {bits_per}")
    total = int(values.size)
    out = bytearray((total * bits_per + 7) // 8)
    mask = (1 << bits_per) - 1
    for idx, value in enumerate(values.flatten()):
        raw = int(value) & mask
        bit_index = idx * bits_per
        byte_index = bit_index // 8
        shift = bit_index % 8
        out[byte_index] |= (raw << shift) & 0xFF
        if shift + bits_per > 8:
            out[byte_index + 1] |= (raw >> (8 - shift)) & 0xFF
    return bytes(out)


def unpack_signed_bits(raw: bytes, bits_per: int, count: int) -> np.ndarray:
    """Unpack signed integers from a compact bitstream."""
    out = np.empty(count, dtype=np.int8)
    mask = (1 << bits_per) - 1
    for idx in range(count):
        bit_index = idx * bits_per
        byte_index = bit_index // 8
        shift = bit_index % 8
        val = (raw[byte_index] >> shift) & mask
        if shift + bits_per > 8:
            val |= (raw[byte_index + 1] << (8 - shift)) & mask
        sign_bit = 1 << (bits_per - 1)
        if val & sign_bit:
            val = val - (1 << bits_per)
        out[idx] = val
    return out


def unpack_unsigned_bits(raw: bytes, bits_per: int, count: int) -> np.ndarray:
    """Unpack unsigned integers from a compact bitstream."""
    out = np.empty(count, dtype=np.uint8)
    mask = (1 << bits_per) - 1
    for idx in range(count):
        bit_index = idx * bits_per
        byte_index = bit_index // 8
        shift = bit_index % 8
        val = (raw[byte_index] >> shift) & mask
        if shift + bits_per > 8:
            val |= (raw[byte_index + 1] << (8 - shift)) & mask
        out[idx] = val
    return out


def unpack_t1_bits(raw: bytes, count: int) -> np.ndarray:
    """Unpack ternary 1-bit values into {-1, 1}."""
    bits = unpack_unsigned_bits(raw, 1, count)
    return np.where(bits == 0, -1, 1).astype(np.int8)
