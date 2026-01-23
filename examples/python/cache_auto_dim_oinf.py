#!/usr/bin/env python3
"""
Create a .oinf file with sizevars for cache auto-dim examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import write_oinf  # noqa: E402


@dataclass
class CacheAutoDimModel:
    D: int
    H: int


def main() -> None:
    model = CacheAutoDimModel(D=2, H=2)
    output = ROOT / "res/models/cache_auto_dim_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
