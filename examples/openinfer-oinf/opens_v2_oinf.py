#!/usr/bin/env python3
"""
Create a .oinf file for the ops_v2 example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, write_oinf  # noqa: E402


@dataclass
class OpsV2Model:
    V: SizeVar
    S: SizeVar
    R: SizeVar
    C: SizeVar


def build_model() -> OpsV2Model:
    return OpsV2Model(
        V=SizeVar(8),
        S=SizeVar(1),
        R=SizeVar(2),
        C=SizeVar(3),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/ops_v2_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
