#!/usr/bin/env python3
"""
Create a .oinf file for the relu example with constant op settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import TensorSpec, write_oinf  # noqa: E402


@dataclass
class ReluModel:
    B: int
    negative_slope: TensorSpec
    clamp_max: TensorSpec


def build_relu() -> ReluModel:
    B = 256
    negative_slope = np.array(0.1, dtype=np.float32)
    clamp_max = np.array(6.0, dtype=np.float32)
    return ReluModel(
        B=B,
        negative_slope=TensorSpec(negative_slope),
        clamp_max=TensorSpec(clamp_max),
    )


def main() -> None:
    model = build_relu()
    output = ROOT / "res/models/relu_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
