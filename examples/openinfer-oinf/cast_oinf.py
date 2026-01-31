#!/usr/bin/env python3
"""
Create a .oinf file for cast example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, write_oinf  # noqa: E402


@dataclass
class CastModel:
    N: SizeVar


def build_model() -> CastModel:
    return CastModel(
        N=SizeVar(8),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/cast_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
