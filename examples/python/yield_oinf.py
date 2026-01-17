#!/usr/bin/env python3
"""
Create a .oinf file for the upcoming CPU yield/await example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dataclass_to_oinf import TensorSpec, write_oinf  # noqa: E402


@dataclass
class YieldModel:
    B: int
    D: int
    x: TensorSpec
    w: TensorSpec
    bias: TensorSpec


def build_yield() -> YieldModel:
    rng = np.random.default_rng(11)
    B = 10
    D = 1024
    x = rng.normal(0.0, 1.0, size=(B, D)).astype(np.float32)
    w = rng.normal(0.0, 0.1, size=(D, D)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, size=(D,)).astype(np.float32)
    return YieldModel(B=B, D=D, x=TensorSpec(x), w=TensorSpec(w), bias=TensorSpec(bias))


def main() -> None:
    model = build_yield()
    output = ROOT / "res/yield_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
