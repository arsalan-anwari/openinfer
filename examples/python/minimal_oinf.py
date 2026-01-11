#!/usr/bin/env python3
"""
Create a minimal .oinf file with a single sizevar and tensor.
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
class MinimalModel:
    B: int
    a: TensorSpec


def build_minimal() -> MinimalModel:
    rng = np.random.default_rng(1)
    B = 1024
    a = rng.uniform(-1.0, 1.0, size=B).astype(np.float32)
    return MinimalModel(B=B, a=TensorSpec(a))


def main() -> None:
    model = build_minimal()
    output = ROOT / "res/minimal_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
