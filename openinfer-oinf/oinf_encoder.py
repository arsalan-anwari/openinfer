"""Encode Python objects into the Open Infer Neural Format (.oinf)."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Annotated, get_args, get_origin, get_type_hints

import numpy as np

from oinf_common import OinfError, align_up, check_key, encode_string
from oinf_numeric import float_to_bf16_bits, float_to_f8_bits
from oinf_packed import pack_signed_bits, pack_unsigned_bits
from oinf_types import (
    PACKED_BITS_PER,
    PACKED_SIGNED,
    PACKED_UNSIGNED,
    ValueType,
    VT_TO_DTYPE,
    dtype_from_alias,
    value_type_from_numpy_dtype,
)

try:
    from pydantic import TypeAdapter  # pydantic v2
    _HAVE_TYPE_ADAPTER = True
except Exception:
    _HAVE_TYPE_ADAPTER = False

try:
    from pydantic.dataclasses import dataclass as pydantic_dataclass  # v1/v2
except Exception:  # pragma: no cover
    pydantic_dataclass = None


class Bitset:
    """Packed bitset payload."""

    def __init__(self, bits: Iterable[bool]):
        """Create a bitset from an iterable of booleans."""
        self._bits = [bool(b) for b in bits]

    @property
    def bit_count(self) -> int:
        """Return the number of bits in the payload."""
        return len(self._bits)

    def to_packed(self) -> bytes:
        """Pack the bitset into bytes."""
        byte_count = (self.bit_count + 7) // 8
        data = bytearray(byte_count)
        for i, bit in enumerate(self._bits):
            if bit:
                data[i // 8] |= 1 << (i % 8)
        return bytes(data)


class TensorSpec:
    """Represents a tensor value."""

    def __init__(
        self,
        data: Union[np.ndarray, Sequence[Any]],
        dtype: Optional[Union[str, int]] = None,
        name: Optional[str] = None,
    ):
        """Define a tensor with optional dtype override and name."""
        self.data = data
        self.dtype = dtype
        self.name = name


class ScalarValue:
    """Represents a scalar metadata value."""

    def __init__(self, value: Any, dtype: Optional[Union[str, int]] = None):
        """Create a scalar metadata entry with optional dtype override."""
        self.value = value
        self.dtype = dtype


class SizeVar:
    """Represents a size variable value."""

    def __init__(self, value: Any):
        """Create a size variable entry."""
        self.value = value


class UninitializedTensor:
    """Represents a tensor declaration without data."""

    def __init__(self, dtype: Union[str, int], shape: Sequence[int], name: Optional[str] = None):
        """Declare a tensor entry without attached data."""
        self.dtype = dtype
        self.shape = tuple(int(x) for x in shape)
        self.name = name


def _infer_int_type(value: int) -> int:
    if value >= 0:
        if value <= np.iinfo(np.uint8).max:
            return ValueType.U8
        if value <= np.iinfo(np.uint16).max:
            return ValueType.U16
        if value <= np.iinfo(np.uint32).max:
            return ValueType.U32
        return ValueType.U64
    if np.iinfo(np.int8).min <= value <= np.iinfo(np.int8).max:
        return ValueType.I8
    if np.iinfo(np.int16).min <= value <= np.iinfo(np.int16).max:
        return ValueType.I16
    if np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
        return ValueType.I32
    return ValueType.I64


def _validate_dataclass_instance(instance: Any, cls: type) -> Any:
    if dataclasses.is_dataclass(cls):
        if pydantic_dataclass is None:
            raise OinfError("pydantic.dataclasses.dataclass not available; cannot validate dataclass")
        pyd_cls = pydantic_dataclass(cls, config={"arbitrary_types_allowed": True})
        if isinstance(instance, cls):
            data = dataclasses.asdict(instance)
            return pyd_cls(**data)
        if isinstance(instance, dict):
            return pyd_cls(**instance)
        return pyd_cls(instance)
    if _HAVE_TYPE_ADAPTER:
        adapter = TypeAdapter(cls)
        return adapter.validate_python(instance)
    return instance


def _extract_annotated_metadata(annotation: Any, field_metadata: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    origin = get_origin(annotation)
    if origin is not None and origin is Annotated:
        args = get_args(annotation)
        for extra in args[1:]:
            if isinstance(extra, str):
                meta[extra] = True
            elif isinstance(extra, dict):
                meta.update(extra)
    if field_metadata:
        meta.update(field_metadata)
    return meta


def _base_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        return get_args(annotation)[0]
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _as_numpy_array(value: Any, dtype_override: Optional[int]) -> np.ndarray:
    arr = np.array(value)
    if arr.dtype == np.object_:
        raise OinfError("Nested lists must be homogeneous to form an ndarray")
    if dtype_override is not None:
        if dtype_override in (ValueType.BF16, ValueType.F8):
            arr = arr.astype(np.float32, copy=False)
        elif dtype_override in PACKED_SIGNED or dtype_override == ValueType.T1:
            arr = arr.astype(np.int8, copy=False)
        elif dtype_override in PACKED_UNSIGNED:
            arr = arr.astype(np.uint8, copy=False)
        else:
            arr = arr.astype(VT_TO_DTYPE[dtype_override], copy=False)
    return arr


def _encode_scalar(value: Any, value_type: int) -> bytes:
    if value_type == ValueType.BOOL:
        return b"\x01" if bool(value) else b"\x00"
    if value_type == ValueType.STRING:
        if not isinstance(value, str):
            raise OinfError("STRING metadata must be str")
        return encode_string(value)
    if value_type == ValueType.BITSET:
        if not isinstance(value, Bitset):
            raise OinfError("BITSET metadata must be Bitset")
        bits = value.bit_count
        packed = value.to_packed()
        header = struct.pack("<II", bits, len(packed))
        payload = header + packed
        padding = b"\x00" * (align_up(len(payload)) - len(payload))
        return payload + padding
    if value_type == ValueType.NDARRAY:
        raise OinfError("NDARRAY requires separate encoding")
    if value_type == ValueType.BF16:
        bits = float_to_bf16_bits(float(value))
        return struct.pack("<H", bits)
    if value_type == ValueType.F8:
        bits = float_to_f8_bits(float(value))
        return struct.pack("<B", bits)
    if value_type in PACKED_BITS_PER:
        bits_per = PACKED_BITS_PER[value_type]
        raw = int(value)
        if value_type in PACKED_SIGNED:
            min_val = -(1 << (bits_per - 1))
            max_val = (1 << (bits_per - 1)) - 1
            if not (min_val <= raw <= max_val):
                raise OinfError(f"Value out of range for i{bits_per}")
            return pack_signed_bits(np.array([raw], dtype=np.int8), bits_per)
        if value_type in PACKED_UNSIGNED:
            max_val = (1 << bits_per) - 1
            if not (0 <= raw <= max_val):
                raise OinfError(f"Value out of range for u{bits_per}")
            return pack_unsigned_bits(np.array([raw], dtype=np.uint8), bits_per)
        if value_type == ValueType.T1:
            if raw not in (-1, 1):
                raise OinfError("Value out of range for t1")
            mapped = 0 if raw < 0 else 1
            return pack_unsigned_bits(np.array([mapped], dtype=np.uint8), bits_per)
        if value_type == ValueType.T2:
            if raw < -1 or raw > 1:
                raise OinfError("Value out of range for t2")
            return pack_signed_bits(np.array([raw], dtype=np.int8), bits_per)
    dtype = VT_TO_DTYPE[value_type]
    arr = np.array(value, dtype=dtype)
    return arr.astype(np.dtype(dtype).newbyteorder("<")).tobytes()


def _encode_ndarray(value: Any, dtype_override: Optional[int]) -> Tuple[bytes, int]:
    arr = _as_numpy_array(value, dtype_override)
    if dtype_override in (ValueType.BF16, ValueType.F8):
        if dtype_override == ValueType.BF16:
            bits = np.vectorize(float_to_bf16_bits, otypes=[np.uint16])(arr)
            data = bits.astype(np.dtype(np.uint16).newbyteorder("<")).tobytes()
        else:
            bits = np.vectorize(float_to_f8_bits, otypes=[np.uint8])(arr)
            data = bits.astype(np.uint8).tobytes()
        header = struct.pack("<II", dtype_override, arr.ndim)
        dims = struct.pack("<" + "Q" * arr.ndim, *[int(d) for d in arr.shape])
        payload = header + dims + data
        padding = b"\x00" * (align_up(len(payload)) - len(payload))
        return payload + padding, dtype_override
    if dtype_override in PACKED_BITS_PER:
        bits_per = PACKED_BITS_PER[dtype_override]
        if dtype_override in PACKED_SIGNED:
            min_val = -(1 << (bits_per - 1))
            max_val = (1 << (bits_per - 1)) - 1
            if arr.size:
                if int(arr.min()) < min_val or int(arr.max()) > max_val:
                    raise OinfError(f"Values out of range for i{bits_per}")
            data = pack_signed_bits(arr.astype(np.int8), bits_per)
        elif dtype_override in PACKED_UNSIGNED:
            max_val = (1 << bits_per) - 1
            if arr.size:
                if int(arr.min()) < 0 or int(arr.max()) > max_val:
                    raise OinfError(f"Values out of range for u{bits_per}")
            data = pack_unsigned_bits(arr.astype(np.uint8), bits_per)
        elif dtype_override == ValueType.T1:
            if arr.size and (int(arr.min()) < -1 or int(arr.max()) > 1):
                raise OinfError("Values out of range for t1")
            mapped = np.where(arr.astype(np.int8) < 0, 0, 1).astype(np.uint8)
            if arr.size and not np.all(np.isin(arr, (-1, 1))):
                raise OinfError("Values out of range for t1")
            data = pack_unsigned_bits(mapped, bits_per)
        else:
            if arr.size and (int(arr.min()) < -1 or int(arr.max()) > 1):
                raise OinfError("Values out of range for t2")
            data = pack_signed_bits(arr.astype(np.int8), bits_per)
        header = struct.pack("<II", dtype_override, arr.ndim)
        dims = struct.pack("<" + "Q" * arr.ndim, *[int(d) for d in arr.shape])
        payload = header + dims + data
        padding = b"\x00" * (align_up(len(payload)) - len(payload))
        return payload + padding, dtype_override
    if dtype_override is None:
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        elif np.issubdtype(arr.dtype, np.signedinteger):
            min_val = int(arr.min()) if arr.size else 0
            max_val = int(arr.max()) if arr.size else 0
            if np.iinfo(np.int8).min <= min_val <= max_val <= np.iinfo(np.int8).max:
                arr = arr.astype(np.int8)
            elif np.iinfo(np.int16).min <= min_val <= max_val <= np.iinfo(np.int16).max:
                arr = arr.astype(np.int16)
            elif np.iinfo(np.int32).min <= min_val <= max_val <= np.iinfo(np.int32).max:
                arr = arr.astype(np.int32)
            else:
                arr = arr.astype(np.int64)
        elif np.issubdtype(arr.dtype, np.unsignedinteger):
            max_val = int(arr.max()) if arr.size else 0
            if max_val <= np.iinfo(np.uint8).max:
                arr = arr.astype(np.uint8)
            elif max_val <= np.iinfo(np.uint16).max:
                arr = arr.astype(np.uint16)
            elif max_val <= np.iinfo(np.uint32).max:
                arr = arr.astype(np.uint32)
            else:
                arr = arr.astype(np.uint64)
    value_type = value_type_from_numpy_dtype(arr.dtype)
    if value_type in (ValueType.STRING, ValueType.BITSET, ValueType.NDARRAY):
        raise OinfError("NDARRAY must be numeric or bool")
    if value_type == ValueType.BOOL:
        data = arr.astype(np.uint8).tobytes()
    else:
        data = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
    header = struct.pack("<II", value_type, arr.ndim)
    dims = struct.pack("<" + "Q" * arr.ndim, *[int(d) for d in arr.shape])
    payload = header + dims + data
    padding = b"\x00" * (align_up(len(payload)) - len(payload))
    return payload + padding, value_type


def _resolve_dtype(value: Any, meta: Dict[str, Any]) -> Optional[int]:
    if "dtype" in meta:
        alias = meta["dtype"]
        vt = dtype_from_alias(alias)
        if vt is None and isinstance(alias, int):
            vt = alias
        if vt is None:
            raise OinfError(f"Unknown dtype override: {alias}")
        return vt
    if isinstance(value, np.ndarray):
        return value_type_from_numpy_dtype(value.dtype)
    return None


def _sizevar_field_names(fields: Sequence[dataclasses.Field], hints: Dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for field in fields:
        annotation = hints.get(field.name, field.type)
        meta = _extract_annotated_metadata(annotation, field.metadata)
        base_type = _base_type(annotation)
        if meta.get("sizevar") or base_type in (int, SizeVar):
            names.add(field.name)
    return names


def _collect_fields(
    instance: Any,
    fields: Sequence[dataclasses.Field],
    sizevar_names: set[str],
    hints: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, Tuple[Any, Dict[str, Any]]], Dict[str, Tuple[Any, Dict[str, Any]]]]:
    sizevars: Dict[str, int] = {}
    metadata: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    tensors: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    if hasattr(instance, "sizevars") and isinstance(instance.sizevars, dict):
        for key, value in instance.sizevars.items():
            check_key(key)
            if isinstance(value, SizeVar):
                value = value.value
            sizevars[key] = int(value)
    if hasattr(instance, "metadata") and isinstance(instance.metadata, dict):
        for key, value in instance.metadata.items():
            check_key(key)
            metadata[key] = (value, {})
    if hasattr(instance, "tensors") and isinstance(instance.tensors, dict):
        for key, value in instance.tensors.items():
            check_key(key)
            tensors[key] = (value, {})

    for field in fields:
        name = field.name
        if name in ("sizevars", "metadata", "tensors"):
            continue
        check_key(name)
        value = getattr(instance, name)
        annotation = hints.get(name, field.type)
        meta = _extract_annotated_metadata(annotation, field.metadata)
        if isinstance(value, SizeVar):
            if value.value is None:
                raise OinfError(f"Sizevar '{name}' cannot be None")
            if isinstance(value.value, bool):
                raise OinfError(f"Sizevar '{name}' cannot be bool")
            sizevars[name] = int(value.value)
            continue
        if name in sizevar_names:
            if isinstance(value, bool):
                raise OinfError(f"Sizevar '{name}' cannot be bool")
            if value is None:
                raise OinfError(f"Sizevar '{name}' cannot be None")
            sizevars[name] = int(value)
            continue

        if isinstance(value, ScalarValue):
            metadata[name] = (value, meta)
            continue

        dtype_override = _resolve_dtype(value, meta)
        is_tensor = meta.get("tensor") or isinstance(value, (TensorSpec, UninitializedTensor, np.ndarray))
        if isinstance(value, (list, tuple)) and dtype_override is not None:
            is_tensor = True
        if is_tensor:
            tensors[name] = (value, meta)
        else:
            metadata[name] = (value, meta)

    return sizevars, metadata, tensors


def _validate_sizevars(sizevars: Dict[str, int]) -> Dict[str, int]:
    for key, value in sizevars.items():
        check_key(key)
        if value < 0:
            raise OinfError(f"Sizevar '{key}' must be non-negative")
        if value > 2**64 - 1:
            raise OinfError(f"Sizevar '{key}' exceeds u64")
    return sizevars


def _encode_metadata_payload(value: Any, meta: Dict[str, Any]) -> Tuple[int, bytes]:
    if isinstance(value, ScalarValue):
        merged_meta = dict(meta)
        if value.dtype is not None:
            merged_meta["dtype"] = value.dtype
        return _encode_metadata_payload(value.value, merged_meta)
    if isinstance(value, bool):
        return ValueType.BOOL, _encode_scalar(value, ValueType.BOOL)
    if isinstance(value, str):
        return ValueType.STRING, _encode_scalar(value, ValueType.STRING)
    if isinstance(value, Bitset):
        return ValueType.BITSET, _encode_scalar(value, ValueType.BITSET)
    if isinstance(value, (int, np.integer)):
        dtype_override = _resolve_dtype(value, meta)
        value_type = dtype_override or _infer_int_type(int(value))
        return value_type, _encode_scalar(int(value), value_type)
    if isinstance(value, (float, np.floating)):
        dtype_override = _resolve_dtype(value, meta)
        value_type = dtype_override or ValueType.F32
        return value_type, _encode_scalar(float(value), value_type)
    if isinstance(value, (list, tuple, np.ndarray)):
        payload, _ = _encode_ndarray(value, _resolve_dtype(value, meta))
        return ValueType.NDARRAY, payload
    raise OinfError(f"Unsupported metadata type: {type(value)}")


def _encode_tensor_payload(
    value: Any,
    dtype_override: Optional[int],
) -> Tuple[int, Tuple[int, ...], bytes]:
    arr = _as_numpy_array(value, dtype_override)
    if dtype_override in (ValueType.BF16, ValueType.F8):
        if dtype_override == ValueType.BF16:
            bits = np.vectorize(float_to_bf16_bits, otypes=[np.uint16])(arr)
            data = bits.astype(np.dtype(np.uint16).newbyteorder("<")).tobytes()
        else:
            bits = np.vectorize(float_to_f8_bits, otypes=[np.uint8])(arr)
            data = bits.astype(np.uint8).tobytes()
        return dtype_override, tuple(int(d) for d in arr.shape), data
    if dtype_override in PACKED_BITS_PER:
        bits_per = PACKED_BITS_PER[dtype_override]
        if dtype_override in PACKED_SIGNED:
            min_val = -(1 << (bits_per - 1))
            max_val = (1 << (bits_per - 1)) - 1
            if arr.size:
                if int(arr.min()) < min_val or int(arr.max()) > max_val:
                    raise OinfError(f"Values out of range for i{bits_per}")
            data = pack_signed_bits(arr.astype(np.int8), bits_per)
        elif dtype_override in PACKED_UNSIGNED:
            max_val = (1 << bits_per) - 1
            if arr.size:
                if int(arr.min()) < 0 or int(arr.max()) > max_val:
                    raise OinfError(f"Values out of range for u{bits_per}")
            data = pack_unsigned_bits(arr.astype(np.uint8), bits_per)
        elif dtype_override == ValueType.T1:
            if arr.size and not np.all(np.isin(arr, (-1, 1))):
                raise OinfError("Values out of range for t1")
            mapped = np.where(arr.astype(np.int8) < 0, 0, 1).astype(np.uint8)
            data = pack_unsigned_bits(mapped, bits_per)
        else:
            if arr.size and (int(arr.min()) < -1 or int(arr.max()) > 1):
                raise OinfError("Values out of range for t2")
            data = pack_signed_bits(arr.astype(np.int8), bits_per)
        return dtype_override, tuple(int(d) for d in arr.shape), data
    dtype = value_type_from_numpy_dtype(arr.dtype)
    if dtype == ValueType.BOOL:
        data = arr.astype(np.uint8).tobytes()
    else:
        data = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
    return dtype, tuple(int(d) for d in arr.shape), data


def _encode_tensor(name: str, value: Any, meta: Dict[str, Any]) -> Tuple[str, int, Tuple[int, ...], int, Optional[bytes]]:
    tensor_name = name
    if isinstance(value, TensorSpec):
        if value.name:
            tensor_name = value.name
        dtype_override = _resolve_dtype(value.data, {"dtype": value.dtype} if value.dtype else {})
        dtype, shape, data = _encode_tensor_payload(value.data, dtype_override)
        return tensor_name, dtype, shape, 1, data
    if isinstance(value, UninitializedTensor):
        if value.name:
            tensor_name = value.name
        dtype = dtype_from_alias(value.dtype) if isinstance(value.dtype, str) else int(value.dtype)
        if dtype not in VT_TO_DTYPE:
            raise OinfError(f"Unsupported tensor dtype for '{tensor_name}'")
        return tensor_name, dtype, tuple(int(d) for d in value.shape), 0, None
    if isinstance(value, np.ndarray):
        dtype, shape, data = _encode_tensor_payload(value, None)
        return tensor_name, dtype, shape, 1, data
    if isinstance(value, (list, tuple)):
        dtype_override = _resolve_dtype(value, meta)
        dtype, shape, data = _encode_tensor_payload(value, dtype_override)
        return tensor_name, dtype, shape, 1, data
    if value is None:
        dtype_override = _resolve_dtype(value, meta)
        if dtype_override is None:
            raise OinfError(f"Tensor '{tensor_name}' is None but no dtype provided")
        shape = meta.get("shape")
        if shape is None:
            raise OinfError(f"Tensor '{tensor_name}' is None but no shape provided")
        return tensor_name, dtype_override, tuple(int(d) for d in shape), 0, None
    raise OinfError(f"Unsupported tensor type for '{tensor_name}': {type(value)}")


def _build_sizevars_table(sizevars: List[Tuple[str, int]]) -> bytes:
    parts = []
    for name, value in sizevars:
        parts.append(encode_string(name))
        parts.append(struct.pack("<Q", int(value)))
    return b"".join(parts)


def _build_metadata_table(
    metadata: List[Tuple[str, int, bytes]],
    offsets: Optional[List[int]] = None,
) -> bytes:
    parts = []
    for idx, (key, value_type, payload) in enumerate(metadata):
        offset = offsets[idx] if offsets is not None else 0
        parts.append(encode_string(key))
        parts.append(struct.pack("<IIQQ", value_type, 0, len(payload), offset))
    return b"".join(parts)


def _build_tensor_table(
    tensors: List[Tuple[str, int, Tuple[int, ...], int, Optional[bytes]]],
    offsets: Optional[List[int]] = None,
) -> bytes:
    parts = []
    for idx, (name, dtype, dims, has_data, payload) in enumerate(tensors):
        offset = offsets[idx] if offsets is not None else 0
        data_nbytes = 0 if payload is None else len(payload)
        parts.append(encode_string(name))
        parts.append(struct.pack("<III", dtype, len(dims), has_data))
        if dims:
            parts.append(struct.pack("<" + "Q" * len(dims), *dims))
        parts.append(struct.pack("<QQ", data_nbytes, offset))
    return b"".join(parts)


def _encode_oinf(
    sizevars: Dict[str, int],
    metadata: Dict[str, Tuple[Any, Dict[str, Any]]],
    tensors: Dict[str, Tuple[Any, Dict[str, Any]]],
) -> bytes:
    sizevars_list = sorted(sizevars.items(), key=lambda kv: kv[0])
    metadata_entries: List[Tuple[str, int, bytes]] = []
    for key, (value, meta) in sorted(metadata.items(), key=lambda kv: kv[0]):
        payload_type, payload = _encode_metadata_payload(value, meta)
        metadata_entries.append((key, payload_type, payload))
    tensor_entries: List[Tuple[str, int, Tuple[int, ...], int, Optional[bytes]]] = []
    for name, (value, meta) in sorted(tensors.items(), key=lambda kv: kv[0]):
        tensor_entries.append(_encode_tensor(name, value, meta))

    header_size = 69
    header_padded = align_up(header_size)
    sizevars_table = _build_sizevars_table(sizevars_list)
    metadata_table_zero = _build_metadata_table(metadata_entries)
    tensor_table_zero = _build_tensor_table(tensor_entries)

    offset_sizevars = header_padded
    offset_metadata = align_up(offset_sizevars + len(sizevars_table))
    offset_tensors = align_up(offset_metadata + len(metadata_table_zero))
    offset_data = align_up(offset_tensors + len(tensor_table_zero))

    data_cursor = offset_data
    metadata_offsets: List[int] = []
    metadata_blobs: List[bytes] = []
    for _, _, payload in metadata_entries:
        metadata_offsets.append(data_cursor)
        metadata_blobs.append(payload)
        data_cursor += align_up(len(payload))

    tensor_offsets: List[int] = []
    tensor_blobs: List[bytes] = []
    for _, _, _, has_data, payload in tensor_entries:
        if has_data:
            tensor_offsets.append(data_cursor)
            tensor_blobs.append(payload or b"")
            data_cursor += align_up(len(payload or b""))
        else:
            tensor_offsets.append(0)
            tensor_blobs.append(b"")

    metadata_table = _build_metadata_table(metadata_entries, metadata_offsets)
    tensor_table = _build_tensor_table(tensor_entries, tensor_offsets)

    file_size = data_cursor
    header = struct.pack(
        "<5sIIIIIIQQQQQ",
        b"OINF\x00",
        1,
        0,
        len(sizevars_list),
        len(metadata_entries),
        len(tensor_entries),
        0,
        offset_sizevars,
        offset_metadata,
        offset_tensors,
        offset_data,
        file_size,
    )
    header += b"\x00" * (header_padded - len(header))

    output = bytearray()
    output += header
    output += sizevars_table
    output += b"\x00" * (offset_metadata - len(output))
    output += metadata_table
    output += b"\x00" * (offset_tensors - len(output))
    output += tensor_table
    output += b"\x00" * (offset_data - len(output))

    for payload in metadata_blobs:
        output += payload
        output += b"\x00" * (align_up(len(payload)) - len(payload))
    for payload, offset in zip(tensor_blobs, tensor_offsets):
        if offset == 0:
            continue
        output += payload
        output += b"\x00" * (align_up(len(payload)) - len(payload))

    if len(output) != file_size:
        raise OinfError(f"Internal error: file size mismatch {len(output)} != {file_size}")
    return bytes(output)


def dataclass_to_oinf(instance: Any) -> bytes:
    """Encode a dataclass instance into an .oinf binary payload."""
    if not dataclasses.is_dataclass(instance):
        raise OinfError("Input must be a dataclass instance")

    original_fields = dataclasses.fields(type(instance))
    hints = get_type_hints(type(instance), include_extras=True)
    sizevar_names = _sizevar_field_names(original_fields, hints)
    validated = _validate_dataclass_instance(instance, type(instance))
    sizevars, metadata, tensors = _collect_fields(validated, original_fields, sizevar_names, hints)
    sizevars = _validate_sizevars(sizevars)

    return _encode_oinf(sizevars, metadata, tensors)


def _load_instance(target: str, json_path: Optional[str]) -> Any:
    if ":" not in target:
        raise OinfError("Input must be in module:ClassOrInstance format")
    module_name, obj_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    if isinstance(obj, type) and dataclasses.is_dataclass(obj):
        if json_path:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return obj(**data)
        try:
            return obj()
        except Exception as exc:
            raise OinfError(f"Failed to instantiate {obj_name}: {exc}") from exc
    if dataclasses.is_dataclass(obj):
        return obj
    raise OinfError(f"{obj_name} is not a dataclass or dataclass instance")


def write_oinf(instance: Any, output_path: str) -> None:
    """Encode a dataclass instance and write it to a file."""
    payload = dataclass_to_oinf(instance)
    with open(output_path, "wb") as handle:
        handle.write(payload)


def main() -> None:
    """CLI entry point for dataclass-to-.oinf encoding."""
    parser = argparse.ArgumentParser(description="Convert a dataclass instance into .oinf")
    parser.add_argument("--input", required=True, help="module:ClassName or module:instance")
    parser.add_argument("--output", required=True, help="Output .oinf path")
    parser.add_argument("--json", help="JSON data for dataclass constructor")
    args = parser.parse_args()

    instance = _load_instance(args.input, args.json)
    payload = dataclass_to_oinf(instance)
    with open(args.output, "wb") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
