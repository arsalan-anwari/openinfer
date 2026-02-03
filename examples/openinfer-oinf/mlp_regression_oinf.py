#!/usr/bin/env python3
"""
Create a .oinf file for a small MLP regression model.
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
class MlpRegressionModel:
    B: SizeVar
    D: SizeVar
    H: SizeVar
    O: SizeVar
    w1: TensorSpec
    b1: TensorSpec
    w2: TensorSpec
    b2: TensorSpec


def build_model() -> MlpRegressionModel:
    rng = np.random.default_rng(1)
    B, D, H, O = 4, 16, 32, 8
    w1 = rng.normal(scale=0.2, size=(D, H)).astype(np.float32)
    b1 = rng.normal(scale=0.05, size=(H,)).astype(np.float32)
    w2 = rng.normal(scale=0.2, size=(H, O)).astype(np.float32)
    b2 = rng.normal(scale=0.05, size=(O,)).astype(np.float32)
    return MlpRegressionModel(
        B=SizeVar(B),
        D=SizeVar(D),
        H=SizeVar(H),
        O=SizeVar(O),
        w1=TensorSpec(w1),
        b1=TensorSpec(b1),
        w2=TensorSpec(w2),
        b2=TensorSpec(b2),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/mlp_regression.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
