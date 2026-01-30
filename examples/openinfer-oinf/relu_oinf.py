#!/usr/bin/env python3
"""
Create a .oinf file for the relu example with constant op settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import ScalarValue, SizeVar, write_oinf  # noqa: E402


@dataclass
class ReluModel:
    B: SizeVar
    alpha: ScalarValue
    clamp_max: ScalarValue


def build_relu() -> ReluModel:
    B = 256
    return ReluModel(
        B=SizeVar(B),
        alpha=ScalarValue(0.1, dtype="f32"),
        clamp_max=ScalarValue(6.0, dtype="f32"),
    )


def main() -> None:
    model = build_relu()
    output = ROOT / "res/models/relu_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
