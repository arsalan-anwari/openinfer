#!/usr/bin/env python3
"""
Verify and pretty-print an Open Infer Neural Format (.oinf) file.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from typing import Any, Dict, List, Tuple

import numpy as np


_ASCII_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class OinfError(ValueError):
    """Raised for OINF validation errors."""


def _align_up(value: int, alignment: int = 8) -> int:
    return (value + alignment - 1) // alignment * alignment


def _check_key(key: str) -> None:
    if not _ASCII_KEY_RE.match(key):
        raise OinfError(f"Invalid string '{key}': must match [A-Za-z0-9._-]+")


class ValueType:
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


_VT_NAME = {
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
}

_VT_SIZE = {
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
}

_VT_TO_DTYPE = {
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
}


def _read_string(blob: bytes, offset: int) -> Tuple[str, int]:
    if offset + 4 > len(blob):
        raise OinfError("String length exceeds file")
    length = struct.unpack_from("<I", blob, offset)[0]
    start = offset + 4
    end = start + length
    if end > len(blob):
        raise OinfError("String payload exceeds file")
    raw = blob[start:end]
    text = raw.decode("ascii")
    _check_key(text)
    padded = _align_up(4 + length)
    return text, offset + padded


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"\"{value}\""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_values_1d(values: np.ndarray) -> str:
    flat = values.flatten()
    total = flat.size
    if total <= 10:
        items = [_format_scalar(v.item()) for v in flat]
    else:
        first = [_format_scalar(v.item()) for v in flat[:5]]
        last = [_format_scalar(v.item()) for v in flat[-5:]]
        items = first + ["..."] + last
    return "{ " + ", ".join(items) + " }"


def _format_values(values: np.ndarray) -> str:
    if values.ndim <= 1:
        return _format_values_1d(values)
    lines = ["{ "]
    rows = min(values.shape[0], values.ndim, 5)
    for idx in range(rows):
        row = np.array(values[idx]).flatten()
        lines.append(f"{_format_values_1d(row)} ,")
    if values.shape[0] > rows:
        lines.append("...")
    lines.append("}")
    return "\n".join(lines)


def _tensor_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "numel": 0,
            "nbytes": int(values.nbytes),
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
        }
    numeric = values.astype(np.float64) if values.dtype == np.bool_ else values.astype(np.float64, copy=False)
    stats = {
        "numel": int(values.size),
        "nbytes": int(values.nbytes),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "median": float(np.median(numeric)),
        "std": float(np.std(numeric)),
    }
    return stats


def _histogram_lines(values: np.ndarray) -> List[str]:
    if values.size == 0:
        return ["(empty)"]
    if values.dtype == np.bool_:
        counts = np.bincount(values.astype(np.uint8), minlength=2)
        return [f"0:{counts[0]} 1:{counts[1]}"]
    numeric = values.astype(np.float64, copy=False)
    vmin = float(np.min(numeric))
    vmax = float(np.max(numeric))
    if np.issubdtype(values.dtype, np.integer) and vmax - vmin <= 20:
        counts = []
        for val in range(int(vmin), int(vmax) + 1):
            counts.append(f"{val}:{int(np.sum(values == val))}")
        return [" ".join(counts)]
    bins = 10
    if vmin == vmax:
        return [f"[{vmin:.6g}]={values.size}"]
    hist, edges = np.histogram(numeric, bins=bins, range=(vmin, vmax))
    parts = []
    for i in range(bins):
        parts.append(f"[{edges[i]:.6g},{edges[i+1]:.6g}):{int(hist[i])}")
    return parts


def _parse_metadata_value(blob: bytes, entry: Dict[str, Any]) -> Any:
    vtype = entry["value_type"]
    offset = entry["value_offset"]
    nbytes = entry["value_nbytes"]
    if offset + nbytes > len(blob):
        raise OinfError("Metadata value exceeds file bounds")
    payload = blob[offset : offset + nbytes]
    if vtype == ValueType.STRING:
        text, next_offset = _read_string(payload, 0)
        if next_offset != nbytes:
            raise OinfError("STRING payload size mismatch")
        return text
    if vtype == ValueType.BOOL:
        if nbytes != 1:
            raise OinfError("BOOL payload size mismatch")
        return payload[0] != 0
    if vtype in _VT_SIZE:
        expected = _VT_SIZE[vtype]
        if nbytes != expected:
            raise OinfError("Metadata scalar size mismatch")
        dtype = np.dtype(_VT_TO_DTYPE[vtype]).newbyteorder("<")
        return np.frombuffer(payload, dtype=dtype)[0].item()
    if vtype == ValueType.BITSET:
        if nbytes < 8:
            raise OinfError("BITSET payload too small")
        bit_count, byte_count = struct.unpack_from("<II", payload, 0)
        expected_bytes = (bit_count + 7) // 8
        if byte_count != expected_bytes:
            raise OinfError("BITSET byte_count mismatch")
        total = 8 + byte_count
        total = _align_up(total)
        if nbytes != total:
            raise OinfError("BITSET payload size mismatch")
        data = payload[8 : 8 + byte_count]
        return {"bit_count": bit_count, "bytes": data}
    if vtype == ValueType.NDARRAY:
        if nbytes < 8:
            raise OinfError("NDARRAY payload too small")
        element_type, ndim = struct.unpack_from("<II", payload, 0)
        if element_type not in _VT_TO_DTYPE:
            raise OinfError("NDARRAY element type invalid")
        dims_offset = 8
        dims_size = 8 * ndim
        if dims_offset + dims_size > nbytes:
            raise OinfError("NDARRAY dims exceed payload")
        if ndim:
            dims = struct.unpack_from("<" + "Q" * ndim, payload, dims_offset)
        else:
            dims = ()
        data_offset = dims_offset + dims_size
        numel = int(np.prod(dims)) if dims else 1
        elem_size = _VT_SIZE[element_type]
        data_nbytes = numel * elem_size
        total = _align_up(data_offset + data_nbytes)
        if nbytes != total:
            raise OinfError("NDARRAY payload size mismatch")
        raw = payload[data_offset : data_offset + data_nbytes]
        if element_type == ValueType.BOOL:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.bool_)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(_VT_TO_DTYPE[element_type]).newbyteorder("<"))
        return {"dtype": element_type, "dims": dims, "array": arr.reshape(dims)}
    raise OinfError(f"Unsupported metadata value type {vtype}")


def _parse_file(path: str) -> None:
    with open(path, "rb") as handle:
        blob = handle.read()

    if len(blob) < 69:
        raise OinfError("File too small for header")
    header = struct.unpack_from("<5sIIIIIIQQQQQ", blob, 0)
    magic = header[0]
    if magic != b"OINF\x00":
        raise OinfError("Bad magic")
    version = header[1]
    if version != 1:
        raise OinfError(f"Unsupported version {version}")
    n_sizevars, n_metadata, n_tensors = header[3], header[4], header[5]
    offset_sizevars, offset_metadata, offset_tensors, offset_data, file_size = header[7:]
    if file_size != len(blob):
        raise OinfError("File size mismatch")
    offsets = [offset_sizevars, offset_metadata, offset_tensors, offset_data, file_size]
    if offsets != sorted(offsets):
        raise OinfError("Offsets are not ascending")
    for off in offsets[:-1]:
        if off % 8 != 0:
            raise OinfError("Section offset not 8-byte aligned")
        if off > file_size:
            raise OinfError("Section offset exceeds file size")

    cursor = offset_sizevars
    sizevars = []
    names = set()
    for _ in range(n_sizevars):
        name, cursor = _read_string(blob, cursor)
        if name in names:
            raise OinfError(f"Duplicate sizevar '{name}'")
        names.add(name)
        if cursor + 8 > offset_metadata:
            raise OinfError("Sizevars table exceeds metadata offset")
        value = struct.unpack_from("<Q", blob, cursor)[0]
        cursor += 8
        sizevars.append((name, value))

    cursor = offset_metadata
    metadata_entries = []
    names = set()
    for _ in range(n_metadata):
        key, cursor = _read_string(blob, cursor)
        if key in names:
            raise OinfError(f"Duplicate metadata key '{key}'")
        names.add(key)
        if cursor + 24 > offset_tensors:
            raise OinfError("Metadata table exceeds tensor offset")
        value_type, flags, value_nbytes, value_offset = struct.unpack_from("<IIQQ", blob, cursor)
        cursor += 24
        if flags != 0:
            raise OinfError("Metadata flags must be 0")
        if value_offset % 8 != 0:
            raise OinfError("Metadata value offset not aligned")
        if value_offset < offset_data:
            raise OinfError("Metadata value offset precedes data section")
        if value_offset + value_nbytes > file_size:
            raise OinfError("Metadata value exceeds file size")
        if value_type not in _VT_NAME:
            raise OinfError(f"Metadata value type {value_type} invalid")
        metadata_entries.append(
            {
                "key": key,
                "value_type": value_type,
                "value_nbytes": value_nbytes,
                "value_offset": value_offset,
            }
        )

    cursor = offset_tensors
    tensors = []
    names = set()
    for _ in range(n_tensors):
        name, cursor = _read_string(blob, cursor)
        if name in names:
            raise OinfError(f"Duplicate tensor name '{name}'")
        names.add(name)
        if cursor + 12 > offset_data:
            raise OinfError("Tensor table exceeds data offset")
        dtype, ndim, flags = struct.unpack_from("<III", blob, cursor)
        cursor += 12
        if dtype not in _VT_SIZE or dtype == ValueType.BITSET or dtype == ValueType.STRING or dtype == ValueType.NDARRAY:
            raise OinfError("Invalid tensor dtype")
        dims = ()
        if ndim > 0:
            dims = struct.unpack_from("<" + "Q" * ndim, blob, cursor)
            cursor += 8 * ndim
        data_nbytes, data_offset = struct.unpack_from("<QQ", blob, cursor)
        cursor += 16
        has_data = flags & 1
        if not has_data:
            if data_nbytes != 0 or data_offset != 0:
                raise OinfError("Tensor without data must have zero offset/size")
        else:
            if data_offset % 8 != 0:
                raise OinfError("Tensor data offset not aligned")
            if data_offset < offset_data:
                raise OinfError("Tensor data offset precedes data section")
            numel = int(np.prod(dims)) if dims else 1
            expected = numel * _VT_SIZE[dtype]
            if data_nbytes != expected:
                raise OinfError("Tensor data_nbytes mismatch")
            if data_offset + data_nbytes > file_size:
                raise OinfError("Tensor data exceeds file size")
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "ndim": ndim,
                "dims": dims,
                "has_data": bool(has_data),
                "data_nbytes": data_nbytes,
                "data_offset": data_offset,
            }
        )

    for name, value in sizevars:
        print(f"{name} := {value}")
    if sizevars:
        print()

    for entry in metadata_entries:
        value = _parse_metadata_value(blob, entry)
        key = entry["key"]
        vtype = entry["value_type"]
        if vtype == ValueType.NDARRAY:
            dtype = _VT_NAME[value["dtype"]]
            dims = ", ".join(str(d) for d in value["dims"])
            arr = value["array"]
            print(f"{key}: {dtype}[{dims}] = {_format_values(arr)}")
        elif vtype == ValueType.BITSET:
            bit_count = value["bit_count"]
            data = value["bytes"]
            preview = " ".join(f"{b:02x}" for b in data[:4])
            print(f"{key}: bitset[{bit_count}] = {preview}")
        else:
            print(f"{key}: {_VT_NAME[vtype]} = {_format_scalar(value)}")

    if metadata_entries:
        print()

    for tensor in tensors:
        name = tensor["name"]
        dtype = _VT_NAME[tensor["dtype"]]
        dims = ", ".join(str(d) for d in tensor["dims"])
        if tensor["has_data"]:
            data_offset = tensor["data_offset"]
            data_nbytes = tensor["data_nbytes"]
            raw = blob[data_offset : data_offset + data_nbytes]
            if tensor["dtype"] == ValueType.BOOL:
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.bool_)
            else:
                arr = np.frombuffer(raw, dtype=np.dtype(_VT_TO_DTYPE[tensor["dtype"]]).newbyteorder("<"))
            arr = arr.reshape(tensor["dims"]) if tensor["dims"] else arr.reshape(())
            if arr.size == 0:
                print(f"{name}: {dtype}[{dims}] -- uninitialized")
                print()
                continue
            if arr.size == 1:
                print(f"{name}: {dtype} = {_format_scalar(arr.item())}")
                print()
                continue
            print(f"{name}: {dtype}[{dims}] = {_format_values(arr)}")
            stats = _tensor_stats(arr)
            print(
                f"- [nbytes: {stats['nbytes']}, min: {stats['min']:.6g}, max: {stats['max']:.6g}, "
                f"mean: {stats['mean']:.6g}, median: {stats['median']:.6g}, std: {stats['std']:.6g}]"
            )
            print("- hist:")
            for line in _histogram_lines(arr):
                print(f"\t{line}")
            print()
        else:
            print(f"{name}: {dtype}[{dims}] -- uninitialized")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and pretty-print .oinf")
    parser.add_argument("path", help="Path to .oinf file")
    args = parser.parse_args()
    _parse_file(args.path)


if __name__ == "__main__":
    main()
