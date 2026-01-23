#!/usr/bin/env python3
"""
Convert a Python dataclass instance into an Open Infer Neural Format (.oinf) file.
"""

from oinf_encoder import (
    Bitset,
    TensorSpec,
    UninitializedTensor,
    dataclass_to_oinf,
    main,
    write_oinf,
)

__all__ = [
    "Bitset",
    "TensorSpec",
    "UninitializedTensor",
    "dataclass_to_oinf",
    "write_oinf",
]


if __name__ == "__main__":
    main()
