#!/usr/bin/env python3
"""
Create a .oinf file for a multi-head linear-attention-style block.
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
class LinearAttentionModel:
    B: SizeVar
    D: SizeVar
    num_heads: SizeVar
    w_out: TensorSpec
    tensors: dict


def build_model() -> LinearAttentionModel:
    rng = np.random.default_rng(3)
    B, D, num_heads = 4, 32, 2
    tensors: dict[str, TensorSpec] = {}
    for head in range(num_heads):
        tensors[f"attn.wq.{head}"] = TensorSpec(
            rng.normal(scale=0.2, size=(D, D)).astype(np.float32)
        )
        tensors[f"attn.wk.{head}"] = TensorSpec(
            rng.normal(scale=0.2, size=(D, D)).astype(np.float32)
        )
        tensors[f"attn.wv.{head}"] = TensorSpec(
            rng.normal(scale=0.2, size=(D, D)).astype(np.float32)
        )
    w_out = rng.normal(scale=0.2, size=(D, D)).astype(np.float32)
    return LinearAttentionModel(
        B=SizeVar(B),
        D=SizeVar(D),
        num_heads=SizeVar(num_heads),
        w_out=TensorSpec(w_out),
        tensors=tensors,
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/linear_attention.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
