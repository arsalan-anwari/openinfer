#!/usr/bin/env python3
"""
Create a .oinf file with nested loop prefix-table tensors.
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
class LoopModel:
    D: SizeVar
    num_layers: SizeVar
    num_heads: SizeVar
    tensors: dict


def build_loop_model() -> LoopModel:
    rng = np.random.default_rng(7)
    D = 4
    num_layers = 2
    num_heads = 3
    tensors = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"attn.{head}.qkv.{layer}"
            data = rng.normal(size=(D, 3 * D)).astype(np.float32)
            tensors[key] = TensorSpec(data)
    return LoopModel(
        D=SizeVar(D),
        num_layers=SizeVar(num_layers),
        num_heads=SizeVar(num_heads),
        tensors=tensors,
    )


def main() -> None:
    model = build_loop_model()
    output = ROOT / "res/models/loop_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
