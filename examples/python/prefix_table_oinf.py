#!/usr/bin/env python3
"""
Create a .oinf file with multiple prefix table patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dataclass_to_oinf import TensorSpec, write_oinf  # noqa: E402


@dataclass
class PrefixTableModel:
    D: int
    tensors: dict


def build_prefix_table() -> PrefixTableModel:
    rng = np.random.default_rng(123)
    D = 8
    tensors = {}
    for idx in range(11):
        tensors[f"W.{idx}"] = TensorSpec(rng.normal(size=D).astype(np.float32))
    for layer in range(2):
        for head in range(3):
            key = f"QKV.{layer}.{head}"
            tensors[key] = TensorSpec(rng.normal(size=D).astype(np.float32))
    return PrefixTableModel(D=D, tensors=tensors)


def main() -> None:
    model = build_prefix_table()
    output = ROOT / "res/prefix_table_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
