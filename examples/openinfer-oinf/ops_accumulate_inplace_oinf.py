#!/usr/bin/env python3
"""
Create a .oinf file for the full-changes ops example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, write_oinf  # noqa: E402


@dataclass
class OpsFullChangesModel:
    V: SizeVar
    M: SizeVar
    K: SizeVar
    N: SizeVar
    B: SizeVar


def build_model() -> OpsFullChangesModel:
    return OpsFullChangesModel(
        V=SizeVar(8),
        M=SizeVar(2),
        K=SizeVar(3),
        N=SizeVar(3),
        B=SizeVar(2),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/ops_accumulate_inplace_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
