#!/usr/bin/env python3
"""
Create a .oinf file for a streaming pipeline example.
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
class StreamingPipelineModel:
    B: SizeVar
    D: SizeVar
    w: TensorSpec
    bias: TensorSpec


def build_model() -> StreamingPipelineModel:
    rng = np.random.default_rng(5)
    B, D = 4, 16
    w = rng.normal(scale=0.2, size=(D, D)).astype(np.float32)
    bias = rng.normal(scale=0.05, size=(D,)).astype(np.float32)
    return StreamingPipelineModel(
        B=SizeVar(B),
        D=SizeVar(D),
        w=TensorSpec(w),
        bias=TensorSpec(bias),
    )


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/streaming_pipeline.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
