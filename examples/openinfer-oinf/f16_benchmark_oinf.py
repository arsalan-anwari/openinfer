#!/usr/bin/env python3
"""
Create a .oinf file for f16 benchmark sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import SizeVar, write_oinf  # noqa: E402


@dataclass
class F16BenchmarkModel:
    N0: SizeVar
    N1: SizeVar
    N2: SizeVar
    N3: SizeVar


def main() -> None:
    model = F16BenchmarkModel(
        N0=SizeVar(256 * 1024),
        N1=SizeVar(1024 * 1024),
        N2=SizeVar(4096 * 1024),
        N3=SizeVar(16384 * 1024),
    )
    output = ROOT / "res/models/f16_benchmark_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
