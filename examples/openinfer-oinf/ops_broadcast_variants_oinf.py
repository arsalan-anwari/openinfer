#!/usr/bin/env python3
"""
Create a .oinf file for broadcast/inplace/accumulate variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, write_oinf  # noqa: E402


@dataclass
class OpsBroadcastVariantsModel:
    V: SizeVar
    S: SizeVar
    M: SizeVar
    K: SizeVar
    N: SizeVar
    B: SizeVar


def build_model() -> OpsBroadcastVariantsModel:
    return OpsBroadcastVariantsModel(
        V=SizeVar(8),
        S=SizeVar(1),
        M=SizeVar(2),
        K=SizeVar(3),
        N=SizeVar(3),
        B=SizeVar(2),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/ops_broadcast_variants_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
