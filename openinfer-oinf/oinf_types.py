"""Type tables and conversions for the .oinf format."""
from __future__ import annotations

from typing import Optional

import numpy as np

from oinf_common import OinfError


class ValueType:
    """Numeric value type identifiers for .oinf payloads."""
    I8 = 1
    I16 = 2
    I32 = 3
    I64 = 4
    U8 = 5
    U16 = 6
    U32 = 7
    U64 = 8
    F16 = 9
    F32 = 10
    F64 = 11
    BOOL = 12
    BITSET = 13
    STRING = 14
    NDARRAY = 15
    BF16 = 16
    F8 = 17
    I4 = 18
    I2 = 19
    I1 = 20
    U4 = 21
    U2 = 22
    U1 = 23
    T2 = 24
    T1 = 25


VT_NAME = {
    ValueType.I8: "i8",
    ValueType.I16: "i16",
    ValueType.I32: "i32",
    ValueType.I64: "i64",
    ValueType.U8: "u8",
    ValueType.U16: "u16",
    ValueType.U32: "u32",
    ValueType.U64: "u64",
    ValueType.F16: "f16",
    ValueType.F32: "f32",
    ValueType.F64: "f64",
    ValueType.BOOL: "bool",
    ValueType.BITSET: "bitset",
    ValueType.STRING: "str",
    ValueType.NDARRAY: "ndarray",
    ValueType.BF16: "bf16",
    ValueType.F8: "f8",
    ValueType.I4: "i4",
    ValueType.I2: "i2",
    ValueType.I1: "i1",
    ValueType.U4: "u4",
    ValueType.U2: "u2",
    ValueType.U1: "u1",
    ValueType.T2: "t2",
    ValueType.T1: "t1",
}

VT_SIZE = {
    ValueType.I8: 1,
    ValueType.I16: 2,
    ValueType.I32: 4,
    ValueType.I64: 8,
    ValueType.U8: 1,
    ValueType.U16: 2,
    ValueType.U32: 4,
    ValueType.U64: 8,
    ValueType.F16: 2,
    ValueType.F32: 4,
    ValueType.F64: 8,
    ValueType.BOOL: 1,
    ValueType.BF16: 2,
    ValueType.F8: 1,
    ValueType.I4: 1,
    ValueType.I2: 1,
    ValueType.I1: 1,
    ValueType.U4: 1,
    ValueType.U2: 1,
    ValueType.U1: 1,
    ValueType.T2: 1,
    ValueType.T1: 1,
}

VT_TO_DTYPE = {
    ValueType.I8: np.int8,
    ValueType.I16: np.int16,
    ValueType.I32: np.int32,
    ValueType.I64: np.int64,
    ValueType.U8: np.uint8,
    ValueType.U16: np.uint16,
    ValueType.U32: np.uint32,
    ValueType.U64: np.uint64,
    ValueType.F16: np.float16,
    ValueType.F32: np.float32,
    ValueType.F64: np.float64,
    ValueType.BOOL: np.bool_,
    ValueType.BF16: np.uint16,
    ValueType.F8: np.uint8,
    ValueType.I4: np.int8,
    ValueType.I2: np.int8,
    ValueType.I1: np.int8,
    ValueType.U4: np.uint8,
    ValueType.U2: np.uint8,
    ValueType.U1: np.uint8,
    ValueType.T2: np.int8,
    ValueType.T1: np.int8,
}

DTYPE_TO_VT = {
    np.dtype(np.int8): ValueType.I8,
    np.dtype(np.int16): ValueType.I16,
    np.dtype(np.int32): ValueType.I32,
    np.dtype(np.int64): ValueType.I64,
    np.dtype(np.uint8): ValueType.U8,
    np.dtype(np.uint16): ValueType.U16,
    np.dtype(np.uint32): ValueType.U32,
    np.dtype(np.uint64): ValueType.U64,
    np.dtype(np.float16): ValueType.F16,
    np.dtype(np.float32): ValueType.F32,
    np.dtype(np.float64): ValueType.F64,
    np.dtype(np.bool_): ValueType.BOOL,
}

DTYPE_ALIAS = {
    "i8": ValueType.I8,
    "i16": ValueType.I16,
    "i32": ValueType.I32,
    "i64": ValueType.I64,
    "u8": ValueType.U8,
    "u16": ValueType.U16,
    "u32": ValueType.U32,
    "u64": ValueType.U64,
    "f16": ValueType.F16,
    "bf16": ValueType.BF16,
    "f8": ValueType.F8,
    "f8e5m2": ValueType.F8,
    "float8e5m2": ValueType.F8,
    "f32": ValueType.F32,
    "f64": ValueType.F64,
    "bool": ValueType.BOOL,
    "i4": ValueType.I4,
    "i2": ValueType.I2,
    "i1": ValueType.I1,
    "u4": ValueType.U4,
    "u2": ValueType.U2,
    "u1": ValueType.U1,
    "t2": ValueType.T2,
    "t1": ValueType.T1,
}

PACKED_BITS_PER = {
    ValueType.I4: 4,
    ValueType.I2: 2,
    ValueType.I1: 1,
    ValueType.U4: 4,
    ValueType.U2: 2,
    ValueType.U1: 1,
    ValueType.T2: 2,
    ValueType.T1: 1,
}

PACKED_SIGNED = {ValueType.I4, ValueType.I2, ValueType.I1, ValueType.T2}
PACKED_UNSIGNED = {ValueType.U4, ValueType.U2, ValueType.U1}


def dtype_from_alias(alias: Optional[str]) -> Optional[int]:
    """Resolve a dtype alias string to a ValueType code."""
    if alias is None:
        return None
    if isinstance(alias, str):
        key = alias.strip().lower()
        if key in DTYPE_ALIAS:
            return DTYPE_ALIAS[key]
    return None


def value_type_from_numpy_dtype(dtype: np.dtype) -> int:
    """Convert a numpy dtype to a ValueType code."""
    dtype = np.dtype(dtype)
    if dtype not in DTYPE_TO_VT:
        raise OinfError(f"Unsupported numpy dtype {dtype}")
    return DTYPE_TO_VT[dtype]


def tensor_nbytes(vtype: int, numel: int) -> int:
    """Return the byte size for a tensor with numel elements."""
    if vtype in PACKED_BITS_PER:
        bits_per = PACKED_BITS_PER[vtype]
        return (numel * bits_per + 7) // 8
    return numel * VT_SIZE[vtype]
