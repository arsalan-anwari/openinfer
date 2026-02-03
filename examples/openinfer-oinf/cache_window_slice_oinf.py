#!/usr/bin/env python3
"""
Create a .oinf file for a cache window slicing example.
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
class CacheWindowSliceModel:
    D: SizeVar
    limit: TensorSpec
    zero: TensorSpec


def build_model() -> CacheWindowSliceModel:
    D = 8
    limit = np.array(6, dtype=np.int32)
    zero = np.array(0, dtype=np.int32)
    return CacheWindowSliceModel(
        D=SizeVar(D),
        limit=TensorSpec(limit),
        zero=TensorSpec(zero),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/cache_window_slice.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
