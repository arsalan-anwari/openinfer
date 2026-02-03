from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OINF = ROOT / "openinfer-oinf"
sys.path.insert(0, str(OINF))

from oinf_common import OinfError
from oinf_types import ValueType, dtype_from_alias, tensor_nbytes, value_type_from_numpy_dtype
from oinf_numeric import bf16_to_f32, float_to_bf16_bits, float_to_f8_bits, f8_to_f32, f8_to_f32_scalar


class TestTypesNumeric(unittest.TestCase):
    def test_dtype_from_alias(self) -> None:
        self.assertEqual(dtype_from_alias("f8e5m2"), ValueType.F8)
        self.assertEqual(dtype_from_alias("float8e5m2"), ValueType.F8)
        self.assertIsNone(dtype_from_alias("unknown"))

    def test_value_type_from_numpy_dtype(self) -> None:
        self.assertEqual(value_type_from_numpy_dtype(np.dtype(np.bool_)), ValueType.BOOL)
        with self.assertRaises(OinfError):
            value_type_from_numpy_dtype(np.dtype(np.object_))

    def test_tensor_nbytes_packed(self) -> None:
        self.assertEqual(tensor_nbytes(ValueType.I4, 1), 1)
        self.assertEqual(tensor_nbytes(ValueType.I4, 2), 1)
        self.assertEqual(tensor_nbytes(ValueType.I4, 3), 2)

    def test_bf16_roundtrip(self) -> None:
        bits = float_to_bf16_bits(1.5)
        arr = np.array([bits], dtype=np.uint16)
        value = bf16_to_f32(arr)[0]
        self.assertAlmostEqual(float(value), 1.5, places=3)

    def test_f8_roundtrip(self) -> None:
        bits = float_to_f8_bits(2.0)
        arr = np.array([bits], dtype=np.uint8)
        value = f8_to_f32(arr)[0]
        self.assertAlmostEqual(float(value), 2.0, places=1)
        self.assertEqual(f8_to_f32_scalar(float_to_f8_bits(float("inf"))), float("inf"))
