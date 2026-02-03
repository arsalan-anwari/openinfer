#!/usr/bin/env python3
"""
Create a .oinf file for a KV cache decode example.
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
class KvCacheDecodeModel:
    D: SizeVar
    zero: TensorSpec


def build_model() -> KvCacheDecodeModel:
    D = 16
    zero = np.zeros((D,), dtype=np.float32)
    return KvCacheDecodeModel(D=SizeVar(D), zero=TensorSpec(zero))


def main() -> None:
    model = build_model()
    output = ROOT / "res/models/kv_cache_decode.oinf"
    write_oinf(model, str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
