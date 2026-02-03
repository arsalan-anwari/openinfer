#!/usr/bin/env python3
"""
Create a .oinf file for a simple MoE routing example.
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
class MoeRoutingModel:
    B: SizeVar
    D: SizeVar
    O: SizeVar
    w_gate: TensorSpec
    w0: TensorSpec
    b0: TensorSpec
    w1: TensorSpec
    b1: TensorSpec


def build_model() -> MoeRoutingModel:
    rng = np.random.default_rng(4)
    B, D, O = 4, 16, 8
    w_gate = rng.normal(scale=0.2, size=(D, 2)).astype(np.float32)
    w0 = rng.normal(scale=0.2, size=(D, O)).astype(np.float32)
    b0 = rng.normal(scale=0.05, size=(O,)).astype(np.float32)
    w1 = rng.normal(scale=0.2, size=(D, O)).astype(np.float32)
    b1 = rng.normal(scale=0.05, size=(O,)).astype(np.float32)
    return MoeRoutingModel(
        B=SizeVar(B),
        D=SizeVar(D),
        O=SizeVar(O),
        w_gate=TensorSpec(w_gate),
        w0=TensorSpec(w0),
        b0=TensorSpec(b0),
        w1=TensorSpec(w1),
        b1=TensorSpec(b1),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/moe_routing.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
