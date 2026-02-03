from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
OINF = ROOT / "openinfer-oinf"
sys.path.insert(0, str(OINF))

from oinf_common import OinfError, align_up, check_key, encode_string, read_string


class TestCommon(unittest.TestCase):
    def test_align_up(self) -> None:
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), 8)
        self.assertEqual(align_up(8), 8)
        self.assertEqual(align_up(9), 16)

    def test_check_key_validation(self) -> None:
        check_key("valid_key-01.OK")
        with self.assertRaises(OinfError):
            check_key("bad key")
        with self.assertRaises(OinfError):
            check_key("naïve")

    def test_encode_read_string_roundtrip(self) -> None:
        blob = encode_string("hello_world")
        value, offset = read_string(blob, 0)
        self.assertEqual(value, "hello_world")
        self.assertEqual(offset, len(blob))

    def test_read_string_bounds(self) -> None:
        with self.assertRaises(OinfError):
            read_string(b"\x01\x00\x00", 0)
