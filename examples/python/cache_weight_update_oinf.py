#!/usr/bin/env python3
"""
Create a .oinf file with sizevars for cache weight update example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dataclass_to_oinf import write_oinf  # noqa: E402


@dataclass
class CacheWeightUpdateModel:
    D: int


def main() -> None:
    model = CacheWeightUpdateModel(D=8)
    output = ROOT / "res/cache_weight_update_model.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
