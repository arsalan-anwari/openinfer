from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "openinfer-oinf"))

from dataclass_to_oinf import TensorSpec, write_oinf  # noqa: E402


@dataclass
class DtypesModel:
    N: int
    a_f64: TensorSpec
    b_f32: TensorSpec
    c_i64: TensorSpec
    d_i32: TensorSpec
    e_i16: TensorSpec
    f_i8: TensorSpec
    g_u64: TensorSpec
    h_u32: TensorSpec
    i_u16: TensorSpec
    j_u8: TensorSpec
    k_bool: TensorSpec
    l_f16: TensorSpec
    m_bf16: TensorSpec
    n_f8: TensorSpec
    o_i4: TensorSpec
    p_i2: TensorSpec
    q_i1: TensorSpec
    r_u4: TensorSpec
    s_u2: TensorSpec
    t_u1: TensorSpec
    u_t2: TensorSpec
    v_t1: TensorSpec


def build_model() -> DtypesModel:
    n = 8
    rng = np.random.default_rng(0)
    return DtypesModel(
        N=n,
        a_f64=TensorSpec(rng.standard_normal(size=n).astype(np.float64)),
        b_f32=TensorSpec(rng.standard_normal(size=n).astype(np.float32)),
        c_i64=TensorSpec(np.arange(-4, 4, dtype=np.int64)),
        d_i32=TensorSpec(np.arange(-4, 4, dtype=np.int32)),
        e_i16=TensorSpec(np.arange(-4, 4, dtype=np.int16)),
        f_i8=TensorSpec(np.arange(-4, 4, dtype=np.int8)),
        g_u64=TensorSpec(np.arange(0, n, dtype=np.uint64)),
        h_u32=TensorSpec(np.arange(0, n, dtype=np.uint32)),
        i_u16=TensorSpec(np.arange(0, n, dtype=np.uint16)),
        j_u8=TensorSpec(np.arange(0, n, dtype=np.uint8)),
        k_bool=TensorSpec(np.array([True, False, True, False, True, False, True, False])),
        l_f16=TensorSpec(rng.uniform(-2.0, 2.0, size=n).astype(np.float16)),
        m_bf16=TensorSpec(rng.uniform(-2.0, 2.0, size=n).astype(np.float32), dtype="bf16"),
        n_f8=TensorSpec(rng.uniform(-1.0, 1.0, size=n).astype(np.float32), dtype="f8"),
        o_i4=TensorSpec(np.array([-8, -1, 0, 1, 7, -3, 4, 2], dtype=np.int8), dtype="i4"),
        p_i2=TensorSpec(np.array([-2, -1, 0, 1, -2, 1, 0, -1], dtype=np.int8), dtype="i2"),
        q_i1=TensorSpec(np.array([-1, 0, -1, 0, -1, 0, -1, 0], dtype=np.int8), dtype="i1"),
        r_u4=TensorSpec(np.array([0, 1, 2, 3, 4, 5, 14, 15], dtype=np.uint8), dtype="u4"),
        s_u2=TensorSpec(np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8), dtype="u2"),
        t_u1=TensorSpec(np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8), dtype="u1"),
        u_t2=TensorSpec(np.array([-1, 0, 1, -1, 0, 1, -1, 0], dtype=np.int8), dtype="t2"),
        v_t1=TensorSpec(np.array([-1, 1, -1, 1, 1, -1, 1, -1], dtype=np.int8), dtype="t1"),
    )


if __name__ == "__main__":
    write_oinf(build_model(), "res/models/dtypes_model.oinf")
