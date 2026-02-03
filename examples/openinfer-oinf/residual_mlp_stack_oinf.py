#!/usr/bin/env python3
"""
Create a .oinf file for a residual MLP stack.
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
class ResidualMlpStackModel:
    B: SizeVar
    D: SizeVar
    num_layers: SizeVar
    bias0: TensorSpec
    tensors: dict


def build_model() -> ResidualMlpStackModel:
    rng = np.random.default_rng(2)
    B, D, num_layers = 4, 32, 3
    tensors: dict[str, TensorSpec] = {}
    for layer in range(num_layers):
        w = rng.normal(scale=0.15, size=(D, D)).astype(np.float32)
        b = rng.normal(scale=0.05, size=(D,)).astype(np.float32)
        tensors[f"res.mlp.w.{layer}"] = TensorSpec(w)
        tensors[f"res.mlp.b.{layer}"] = TensorSpec(b)
    bias0 = np.zeros((D,), dtype=np.float32)
    return ResidualMlpStackModel(
        B=SizeVar(B),
        D=SizeVar(D),
        num_layers=SizeVar(num_layers),
        bias0=TensorSpec(bias0),
        tensors=tensors,
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/residual_mlp_stack.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
