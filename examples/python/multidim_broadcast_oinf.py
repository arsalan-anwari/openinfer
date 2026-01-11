#!/usr/bin/env python3
"""
Create a multi-dim .oinf file for broadcast-friendly tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dataclass_to_oinf import TensorSpec, UninitializedTensor, write_oinf  # noqa: E402


@dataclass
class MultiDimModel:
    A: int
    B: int
    x: TensorSpec
    y: TensorSpec
    z: TensorSpec
    bias: TensorSpec
    out: UninitializedTensor


def build_model() -> MultiDimModel:
    rng = np.random.default_rng(42)
    A = 2
    B = 4
    x = rng.normal(size=(A, B)).astype(np.float32)
    y = rng.normal(size=(B,)).astype(np.float32)
    z = rng.normal(size=(A, 1)).astype(np.float32)
    bias = np.array(0.25, dtype=np.float32)

    return MultiDimModel(
        A=A,
        B=B,
        x=TensorSpec(x),
        y=TensorSpec(y),
        z=TensorSpec(z),
        bias=TensorSpec(bias),
        out=UninitializedTensor(dtype="f32", shape=(A, B)),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/multidim_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
