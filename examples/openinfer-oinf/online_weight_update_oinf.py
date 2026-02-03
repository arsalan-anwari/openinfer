#!/usr/bin/env python3
"""
Create a .oinf file for an online weight update example.
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
class OnlineWeightUpdateModel:
    B: SizeVar
    D: SizeVar
    O: SizeVar
    zero: TensorSpec


def build_model() -> OnlineWeightUpdateModel:
    B, D, O = 4, 16, 8
    zero = np.zeros((D, O), dtype=np.float32)
    return OnlineWeightUpdateModel(
        B=SizeVar(B),
        D=SizeVar(D),
        O=SizeVar(O),
        zero=TensorSpec(zero),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/online_weight_update.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
