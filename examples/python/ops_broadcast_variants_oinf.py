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

from dataclass_to_oinf import write_oinf  # noqa: E402


@dataclass
class OpsBroadcastVariantsModel:
    V: int
    S: int
    M: int
    K: int
    N: int
    B: int


def build_model() -> OpsBroadcastVariantsModel:
    return OpsBroadcastVariantsModel(V=8, S=1, M=2, K=3, N=3, B=2)


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/ops_broadcast_variants_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
