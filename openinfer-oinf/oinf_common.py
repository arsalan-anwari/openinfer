"""Common helpers for the Open Infer Neural Format (.oinf)."""
from __future__ import annotations

import re
import struct
from typing import Tuple


class OinfError(ValueError):
    """Raised for OINF validation or encoding errors."""


ASCII_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def align_up(value: int, alignment: int = 8) -> int:
    """Round a value up to the next alignment boundary."""
    return (value + alignment - 1) // alignment * alignment


def check_key(key: str) -> None:
    """Validate that a key is ASCII-safe for the .oinf format."""
    if not isinstance(key, str):
        raise OinfError(f"Key must be str, got {type(key)}")
    if not ASCII_KEY_RE.match(key):
        raise OinfError(f"Invalid key '{key}': must match [A-Za-z0-9._-]+")


def encode_string(value: str) -> bytes:
    """Encode a key-safe string with length prefix and padding."""
    check_key(value)
    raw = value.encode("ascii")
    header = struct.pack("<I", len(raw))
    payload = header + raw
    padding = b"\x00" * (align_up(len(payload)) - len(payload))
    return payload + padding


def read_string(blob: bytes, offset: int) -> Tuple[str, int]:
    """Read a length-prefixed ASCII string from a blob."""
    if offset + 4 > len(blob):
        raise OinfError("String length exceeds file")
    length = struct.unpack_from("<I", blob, offset)[0]
    start = offset + 4
    end = start + length
    if end > len(blob):
        raise OinfError("String payload exceeds file")
    raw = blob[start:end]
    text = raw.decode("ascii")
    check_key(text)
    padded = align_up(4 + length)
    return text, offset + padded
