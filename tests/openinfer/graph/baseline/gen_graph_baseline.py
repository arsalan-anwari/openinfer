from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import write_oinf  # noqa: E402


@dataclass
class GraphSimpleBaseline:
    simple_x: np.ndarray
    simple_w: np.ndarray
    simple_b: np.ndarray
    simple_out: np.ndarray


@dataclass
class GraphBranchBaseline:
    branch_x: np.ndarray
    branch_out: np.ndarray


def build_graph_simple() -> GraphSimpleBaseline:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    w = rng.normal(size=(3, 4)).astype(np.float32)
    b = rng.normal(size=(2, 4)).astype(np.float32)
    matmul = x @ w
    add = matmul + b
    out = np.maximum(add, 0.0)
    return GraphSimpleBaseline(simple_x=x, simple_w=w, simple_b=b, simple_out=out)


def build_graph_branch() -> GraphBranchBaseline:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    relu = np.maximum(x, 0.0)
    abs_x = np.abs(x)
    out = relu + abs_x
    return GraphBranchBaseline(branch_x=x, branch_out=out)


def write_manifest(path: Path, tensors: dict[str, np.ndarray]) -> None:
    manifest = {
        "file": path.name,
        "tensors": [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensors.items()
        ],
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    data_dir = ROOT / "tests/openinfer/graph/baseline/data"
    data_dir.mkdir(parents=True, exist_ok=True)

    simple = build_graph_simple()
    simple_path = data_dir / "graph_simple.oinf"
    write_oinf(simple, str(simple_path))
    write_manifest(
        simple_path,
        {name: getattr(simple, name) for name in simple.__dataclass_fields__},
    )

    branch = build_graph_branch()
    branch_path = data_dir / "graph_branch.oinf"
    write_oinf(branch, str(branch_path))
    write_manifest(
        branch_path,
        {name: getattr(branch, name) for name in branch.__dataclass_fields__},
    )

    print(f"Wrote {simple_path}")
    print(f"Wrote {branch_path}")


if __name__ == "__main__":
    main()
