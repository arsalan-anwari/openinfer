"""
Create a large .oinf model for worst-case testing.
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
class WorstCaseModel:
    D: SizeVar
    num_layers: SizeVar
    w: TensorSpec


def build_worst_case() -> WorstCaseModel:
    rng = np.random.default_rng(1234)
    num_layers = 1000
    # 64 MiB tensor: 64 * 1024 * 1024 / 4
    # d = 16_777_216
    d = 1024
    w = TensorSpec(rng.uniform(-1.0, 1.0, size=d).astype(np.float32))
    return WorstCaseModel(D=SizeVar(d), num_layers=SizeVar(num_layers), w=w)


def main() -> None:
    model = build_worst_case()
    output = ROOT / "res/models/worst_case_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
