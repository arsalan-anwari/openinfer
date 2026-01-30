#!/usr/bin/env python3
"""
Create a demo .oinf file matching the sample layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, TensorSpec, UninitializedTensor, write_oinf  # noqa: E402


@dataclass
class ExampleModel:
    D: SizeVar
    B: SizeVar
    a: TensorSpec
    x: TensorSpec
    W_0: TensorSpec
    mode: str
    y: UninitializedTensor
    kernel: TensorSpec


def build_example() -> ExampleModel:
    rng = np.random.default_rng(0)
    D = 128
    B = 1024
    a = rng.normal(size=B).astype(np.float16)
    x = np.array(10.35, dtype=np.float32)
    w0 = rng.normal(size=D).astype(np.float32)
    kernel = rng.integers(0, 256, size=(D, D), dtype=np.uint8)

    return ExampleModel(
        D=SizeVar(D),
        B=SizeVar(B),
        a=TensorSpec(a),
        x=TensorSpec(x),
        W_0=TensorSpec(w0, name="W.0"),
        mode="clamp_up",
        y=UninitializedTensor(dtype="i16", shape=()),
        kernel=TensorSpec(kernel),
    )


def main() -> None:
    model = build_example()
    output = ROOT / "res/models/simple_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
