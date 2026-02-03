#!/usr/bin/env python3
"""
Create a .oinf file for a quantized linear matmul example.
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
class QuantizedLinearModel:
    B: SizeVar
    D: SizeVar
    O: SizeVar
    x: TensorSpec
    w: TensorSpec


def build_model() -> QuantizedLinearModel:
    rng = np.random.default_rng(6)
    B, D, O = 4, 16, 8
    x = rng.integers(-8, 8, size=(B, D), dtype=np.int8)
    w = rng.integers(-8, 8, size=(D, O), dtype=np.int8)
    return QuantizedLinearModel(
        B=SizeVar(B),
        D=SizeVar(D),
        O=SizeVar(O),
        x=TensorSpec(x, dtype="i4"),
        w=TensorSpec(w, dtype="i4"),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/quantized_linear.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
