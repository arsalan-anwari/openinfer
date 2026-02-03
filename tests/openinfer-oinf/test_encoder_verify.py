from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OINF = ROOT / "openinfer-oinf"
sys.path.insert(0, str(OINF))

from oinf_encoder import Bitset, ScalarValue, TensorSpec, UninitializedTensor, dataclass_to_oinf, write_oinf
from oinf_verify import parse_file


@dataclasses.dataclass
class Sample:
    a: np.ndarray
    b: TensorSpec
    c: TensorSpec
    d: TensorSpec
    e: TensorSpec
    f: UninitializedTensor


class TestEncoderVerify(unittest.TestCase):
    def test_dataclass_roundtrip_with_verify(self) -> None:
        sample = Sample(
            a=np.array([[1, 2], [3, 4]], dtype=np.int16),
            b=TensorSpec(np.array([1.25, -2.0], dtype=np.float32), dtype="bf16"),
            c=TensorSpec(np.array([0.0, 2.0], dtype=np.float32), dtype="f8"),
            d=TensorSpec(np.array([-2, -1, 0, 1], dtype=np.int8), dtype="i4"),
            e=TensorSpec(np.array([0, 1, 1, 0], dtype=np.int8), dtype="u1"),
            f=UninitializedTensor("f32", (2, 2)),
        )
        sample.sizevars = {"B": 2}
        sample.metadata = {
            "title": "sample",
            "flag": True,
            "count": ScalarValue(3, dtype="u8"),
            "bitset": Bitset([True, False, True, True]),
            "meta_arr": [1, 2, 3],
        }

        payload = dataclass_to_oinf(sample)
        self.assertGreater(len(payload), 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.oinf"
            write_oinf(sample, str(path))
            parse_file(str(path))
