#!/usr/bin/env python3
"""
Create a .oinf file for attribute resolution example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import ScalarValue, SizeVar, write_oinf  # noqa: E402


@dataclass
class AttrsModel:
    B: SizeVar
    alpha: ScalarValue
    clamp_max: ScalarValue
    rounding_mode: str


def build_model() -> AttrsModel:
    return AttrsModel(
        B=SizeVar(128),
        alpha=ScalarValue(0.1, dtype="f32"),
        clamp_max=ScalarValue(6.0, dtype="f32"),
        rounding_mode="trunc",
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/attrs_from_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
