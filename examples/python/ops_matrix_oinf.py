#!/usr/bin/env python3
"""
Create a .oinf file for the ops-matrix example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import write_oinf  # noqa: E402


@dataclass
class OpsMatrixModel:
    V: int
    M: int
    K: int
    N: int


def build_ops_matrix() -> OpsMatrixModel:
    return OpsMatrixModel(V=8, M=2, K=3, N=4)


def main() -> None:
    model = build_ops_matrix()
    output_cpu = ROOT / "res/models/ops_matrix_model.oinf"
    output_vulkan = ROOT / "res/models/ops_matrix_model_vulkan.oinf"
    write_oinf(model, str(output_cpu))
    write_oinf(model, str(output_vulkan))
    print(f"Wrote {output_cpu}")
    print(f"Wrote {output_vulkan}")


if __name__ == "__main__":
    main()
