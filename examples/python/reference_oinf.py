#!/usr/bin/env python3
"""
Create a .oinf file that exercises @reference mappings in the DSL.
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
class ReferenceModel:
    B: int
    weight: TensorSpec
    bias: TensorSpec
    state: TensorSpec


def build_reference() -> ReferenceModel:
    rng = np.random.default_rng(42)
    B = 128
    weight = rng.normal(0.0, 0.5, size=B).astype(np.float32)
    bias = np.array(0.25, dtype=np.float32)
    state = rng.normal(0.0, 0.2, size=B).astype(np.float32)
    return ReferenceModel(
        B=B,
        weight=TensorSpec(weight, name="weight.0"),
        bias=TensorSpec(bias, name="bias.0"),
        state=TensorSpec(state, name="state.0"),
    )


def main() -> None:
    model = build_reference()
    output = ROOT / "res/reference_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
