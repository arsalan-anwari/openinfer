#!/usr/bin/env python3
"""
Create a .oinf file for the branching control-flow example.
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
class BranchingModel:
    B: SizeVar
    D: SizeVar
    w: TensorSpec


def build_branching() -> BranchingModel:
    rng = np.random.default_rng(7)
    B = 1000
    D = 128
    w = rng.normal(0.0, 0.5, size=(D, D)).astype(np.float32)
    return BranchingModel(B=SizeVar(B), D=SizeVar(D), w=TensorSpec(w))


def main() -> None:
    model = build_branching()
    output = ROOT / "res/models/branching_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
