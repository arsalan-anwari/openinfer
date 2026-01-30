#!/usr/bin/env python3
"""
Create a .oinf file for the accumulate-packed example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, TensorSpec, write_oinf  # noqa: E402


@dataclass
class AccumulatePackedModel:
    N: SizeVar
    a_i8: TensorSpec
    b_i8: TensorSpec
    a_i16: TensorSpec
    b_i16: TensorSpec
    a_i32: TensorSpec
    b_i32: TensorSpec
    a_u8: TensorSpec
    b_u8: TensorSpec
    a_u16: TensorSpec
    b_u16: TensorSpec
    a_u32: TensorSpec
    b_u32: TensorSpec
    a_i4: TensorSpec
    b_i4: TensorSpec
    a_i2: TensorSpec
    b_i2: TensorSpec
    a_i1: TensorSpec
    b_i1: TensorSpec
    a_u4: TensorSpec
    b_u4: TensorSpec
    a_u2: TensorSpec
    b_u2: TensorSpec
    a_u1: TensorSpec
    b_u1: TensorSpec


def build_model() -> AccumulatePackedModel:
    n = 8
    signed_vals = np.array([-4, -2, -1, 0, 1, 2, 3, 4], dtype=np.int32)
    signed_vals_b = np.array([1, 1, 2, 2, -1, -2, 0, 1], dtype=np.int32)
    unsigned_vals = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)
    unsigned_vals_b = np.array([1, 0, 2, 1, 0, 2, 1, 0], dtype=np.uint32)

    i4_vals = np.array([-8, -4, -1, 0, 1, 3, 6, 7], dtype=np.int8)
    i4_vals_b = np.array([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.int8)
    i2_vals = np.array([-2, -1, 0, 1, -2, -1, 0, 1], dtype=np.int8)
    i2_vals_b = np.array([1, 0, -1, -2, 1, 0, -1, -2], dtype=np.int8)
    i1_vals = np.array([-1, 0, -1, 0, -1, 0, -1, 0], dtype=np.int8)
    i1_vals_b = np.array([0, -1, 0, -1, 0, -1, 0, -1], dtype=np.int8)
    u4_vals = np.array([0, 1, 2, 3, 4, 5, 14, 15], dtype=np.uint8)
    u4_vals_b = np.array([1, 2, 3, 4, 0, 1, 2, 3], dtype=np.uint8)
    u2_vals = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
    u2_vals_b = np.array([3, 2, 1, 0, 3, 2, 1, 0], dtype=np.uint8)
    u1_vals = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    u1_vals_b = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)

    return AccumulatePackedModel(
        N=SizeVar(n),
        a_i8=TensorSpec(signed_vals.astype(np.int8)),
        b_i8=TensorSpec(signed_vals_b.astype(np.int8)),
        a_i16=TensorSpec(signed_vals.astype(np.int16)),
        b_i16=TensorSpec(signed_vals_b.astype(np.int16)),
        a_i32=TensorSpec(signed_vals.astype(np.int32)),
        b_i32=TensorSpec(signed_vals_b.astype(np.int32)),
        a_u8=TensorSpec(unsigned_vals.astype(np.uint8)),
        b_u8=TensorSpec(unsigned_vals_b.astype(np.uint8)),
        a_u16=TensorSpec(unsigned_vals.astype(np.uint16)),
        b_u16=TensorSpec(unsigned_vals_b.astype(np.uint16)),
        a_u32=TensorSpec(unsigned_vals.astype(np.uint32)),
        b_u32=TensorSpec(unsigned_vals_b.astype(np.uint32)),
        a_i4=TensorSpec(i4_vals, dtype="i4"),
        b_i4=TensorSpec(i4_vals_b, dtype="i4"),
        a_i2=TensorSpec(i2_vals, dtype="i2"),
        b_i2=TensorSpec(i2_vals_b, dtype="i2"),
        a_i1=TensorSpec(i1_vals, dtype="i1"),
        b_i1=TensorSpec(i1_vals_b, dtype="i1"),
        a_u4=TensorSpec(u4_vals, dtype="u4"),
        b_u4=TensorSpec(u4_vals_b, dtype="u4"),
        a_u2=TensorSpec(u2_vals, dtype="u2"),
        b_u2=TensorSpec(u2_vals_b, dtype="u2"),
        a_u1=TensorSpec(u1_vals, dtype="u1"),
        b_u1=TensorSpec(u1_vals_b, dtype="u1"),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/accumulate_packed_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
