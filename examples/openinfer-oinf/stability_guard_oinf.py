#!/usr/bin/env python3
"""
Create a .oinf file for a stability guard example.
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
class StabilityGuardModel:
    B: SizeVar
    D: SizeVar
    bias: TensorSpec


def build_model() -> StabilityGuardModel:
    rng = np.random.default_rng(7)
    B, D = 4, 16
    bias = rng.normal(scale=0.05, size=(D,)).astype(np.float32)
    return StabilityGuardModel(
        B=SizeVar(B),
        D=SizeVar(D),
        bias=TensorSpec(bias),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/stability_guard.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
