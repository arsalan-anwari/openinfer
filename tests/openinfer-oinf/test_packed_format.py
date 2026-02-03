from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OINF = ROOT / "openinfer-oinf"
sys.path.insert(0, str(OINF))

from oinf_format import format_scalar, format_values, histogram_string
from oinf_packed import pack_signed_bits, pack_unsigned_bits, unpack_signed_bits, unpack_t1_bits, unpack_unsigned_bits


class TestPackedFormat(unittest.TestCase):
    def test_pack_unpack_signed(self) -> None:
        values = np.array([-2, -1, 0, 1], dtype=np.int8)
        packed = pack_signed_bits(values, 2)
        out = unpack_signed_bits(packed, 2, 4)
        np.testing.assert_array_equal(out, values)

    def test_pack_unpack_unsigned(self) -> None:
        values = np.array([0, 1, 2, 3], dtype=np.uint8)
        packed = pack_unsigned_bits(values, 2)
        out = unpack_unsigned_bits(packed, 2, 4)
        np.testing.assert_array_equal(out, values)

    def test_unpack_t1(self) -> None:
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        packed = pack_unsigned_bits(bits, 1)
        out = unpack_t1_bits(packed, 4)
        np.testing.assert_array_equal(out, np.array([-1, 1, -1, 1], dtype=np.int8))

    def test_format_helpers(self) -> None:
        self.assertEqual(format_scalar(True), "true")
        self.assertEqual(format_scalar("x"), "\"x\"")
        values = np.arange(12, dtype=np.int32)
        self.assertIn("...", format_values(values))
        hist = histogram_string(np.array([True, False, True], dtype=np.bool_))
        self.assertIn("([0, 0]", hist)
