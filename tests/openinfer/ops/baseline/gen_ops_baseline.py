from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

from dataclass_to_oinf import TensorSpec, write_oinf  # noqa: E402
from oinf_numeric import bf16_to_f32, f8_to_f32, float_to_bf16_bits, float_to_f8_bits  # noqa: E402


@dataclass
class OpsBasicBaseline:
    add_a: np.ndarray
    add_b: np.ndarray
    add_out: np.ndarray
    sub_a: np.ndarray
    sub_b: np.ndarray
    sub_out: np.ndarray
    mul_a: np.ndarray
    mul_b: np.ndarray
    mul_out: np.ndarray
    div_a: np.ndarray
    div_b: np.ndarray
    div_out: np.ndarray
    relu_x: np.ndarray
    relu_out: np.ndarray
    abs_x: np.ndarray
    abs_out: np.ndarray
    neg_x: np.ndarray
    neg_out: np.ndarray
    min_a: np.ndarray
    min_b: np.ndarray
    min_out: np.ndarray
    max_a: np.ndarray
    max_b: np.ndarray
    max_out: np.ndarray
    clamp_x: np.ndarray
    clamp_out: np.ndarray
    sum_axis_x: np.ndarray
    sum_axis_out: np.ndarray
    mean_axis_x: np.ndarray
    mean_axis_out: np.ndarray


@dataclass
class OpsMatmulBaseline:
    matmul_a: np.ndarray
    matmul_b: np.ndarray
    matmul_out: np.ndarray


@dataclass
class OpsBroadcastBaseline:
    bcast_a: np.ndarray
    bcast_b: np.ndarray
    bcast_add_out: np.ndarray
    bcast_mul_out: np.ndarray


@dataclass
class OpsCompareBaseline:
    cmp_a: np.ndarray
    cmp_b: np.ndarray
    eq_out: np.ndarray
    ne_out: np.ndarray
    lt_out: np.ndarray
    le_out: np.ndarray
    gt_out: np.ndarray
    ge_out: np.ndarray


@dataclass
class OpsBitwiseBaseline:
    bit_a: np.ndarray
    bit_b: np.ndarray
    and_out: np.ndarray
    or_out: np.ndarray
    xor_out: np.ndarray
    not_out: np.ndarray
    shl_out: np.ndarray
    shr_out: np.ndarray
    popcount_out: np.ndarray


@dataclass
class OpsRoundingBaseline:
    round_x: np.ndarray
    floor_out: np.ndarray
    ceil_out: np.ndarray
    round_out: np.ndarray
    trunc_out: np.ndarray
    sign_out: np.ndarray
    recip_out: np.ndarray


@dataclass
class OpsReduceBaseline:
    reduce_x: np.ndarray
    sum_axis_out: np.ndarray
    mean_axis_out: np.ndarray
    prod_axis_out: np.ndarray
    min_axis_out: np.ndarray
    max_axis_out: np.ndarray
    argmax_axis_out: np.ndarray
    argmin_axis_out: np.ndarray


@dataclass
class OpsCastBaseline:
    cast_f32: np.ndarray
    cast_f32_to_i32: np.ndarray
    cast_f32_to_u8: np.ndarray
    cast_i32: np.ndarray
    cast_i32_to_f32: np.ndarray


@dataclass
class OpsAccumulateInplaceBaseline:
    acc_a: np.ndarray
    acc_b: np.ndarray
    acc_out: np.ndarray
    inplace_a: np.ndarray
    inplace_b: np.ndarray
    inplace_out: np.ndarray


@dataclass
class OpsMiscBaseline:
    fma_a: np.ndarray
    fma_b: np.ndarray
    fma_c: np.ndarray
    fma_out: np.ndarray
    floor_div_a: np.ndarray
    floor_div_b: np.ndarray
    floor_div_out: np.ndarray
    rem_a: np.ndarray
    rem_b: np.ndarray
    rem_out: np.ndarray
    is_nan_x: np.ndarray
    is_nan_out: np.ndarray
    is_inf_x: np.ndarray
    is_inf_out: np.ndarray
    is_neg_x: np.ndarray
    is_neg_out: np.ndarray
    is_finite_x: np.ndarray
    is_finite_out: np.ndarray


@dataclass
class OpsFloatSpecialBaseline:
    f16_a: TensorSpec
    f16_b: TensorSpec
    f16_add_out: TensorSpec
    f16_relu_x: TensorSpec
    f16_relu_out: TensorSpec
    bf16_a: TensorSpec
    bf16_b: TensorSpec
    bf16_add_out: TensorSpec
    bf16_relu_x: TensorSpec
    bf16_relu_out: TensorSpec
    f8_a: TensorSpec
    f8_b: TensorSpec
    f8_add_out: TensorSpec
    f8_relu_x: TensorSpec
    f8_relu_out: TensorSpec


@dataclass
class OpsPackedBaseline:
    packed_i4_x: TensorSpec
    packed_i4_argmax_out: np.ndarray
    packed_u2_x: TensorSpec
    packed_u2_argmin_out: np.ndarray


def build_ops_basic() -> OpsBasicBaseline:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    b = rng.normal(size=(2, 3)).astype(np.float32)
    div_b = np.where(np.abs(b) < 0.2, 0.5, b).astype(np.float32)
    relu_x = rng.normal(size=(2, 3)).astype(np.float32)
    abs_x = rng.normal(size=(2, 3)).astype(np.float32)
    neg_x = rng.normal(size=(2, 3)).astype(np.float32)
    min_a = rng.normal(size=(2, 3)).astype(np.float32)
    min_b = rng.normal(size=(2, 3)).astype(np.float32)
    max_a = rng.normal(size=(2, 3)).astype(np.float32)
    max_b = rng.normal(size=(2, 3)).astype(np.float32)
    clamp_x = rng.normal(size=(2, 3)).astype(np.float32)
    sum_axis_x = rng.normal(size=(2, 3)).astype(np.float32)
    mean_axis_x = rng.normal(size=(2, 3)).astype(np.float32)

    return OpsBasicBaseline(
        add_a=a,
        add_b=b,
        add_out=a + b,
        sub_a=a,
        sub_b=b,
        sub_out=a - b,
        mul_a=a,
        mul_b=b,
        mul_out=a * b,
        div_a=a,
        div_b=div_b,
        div_out=a / div_b,
        relu_x=relu_x,
        relu_out=np.maximum(relu_x, 0.0),
        abs_x=abs_x,
        abs_out=np.abs(abs_x),
        neg_x=neg_x,
        neg_out=-neg_x,
        min_a=min_a,
        min_b=min_b,
        min_out=np.minimum(min_a, min_b),
        max_a=max_a,
        max_b=max_b,
        max_out=np.maximum(max_a, max_b),
        clamp_x=clamp_x,
        clamp_out=np.clip(clamp_x, 0.0, 3.0),
        sum_axis_x=sum_axis_x,
        sum_axis_out=sum_axis_x.sum(axis=1, keepdims=True),
        mean_axis_x=mean_axis_x,
        mean_axis_out=mean_axis_x.mean(axis=1, keepdims=True),
    )


def build_ops_matmul() -> OpsMatmulBaseline:
    rng = np.random.default_rng(1)
    left = rng.normal(size=(2, 3)).astype(np.float32)
    right = rng.normal(size=(3, 4)).astype(np.float32)
    return OpsMatmulBaseline(
        matmul_a=left,
        matmul_b=right,
        matmul_out=left @ right,
    )


def build_ops_broadcast() -> OpsBroadcastBaseline:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    b = rng.normal(size=(3,)).astype(np.float32)
    return OpsBroadcastBaseline(
        bcast_a=a,
        bcast_b=b,
        bcast_add_out=a + b,
        bcast_mul_out=a * b,
    )


def build_ops_compare() -> OpsCompareBaseline:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    b = rng.normal(size=(2, 3)).astype(np.float32)
    return OpsCompareBaseline(
        cmp_a=a,
        cmp_b=b,
        eq_out=a == b,
        ne_out=a != b,
        lt_out=a < b,
        le_out=a <= b,
        gt_out=a > b,
        ge_out=a >= b,
    )


def popcount_u8(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, 1)
    counts = np.unpackbits(flat, axis=1).sum(axis=1)
    return counts.astype(np.uint8).reshape(values.shape)


def build_ops_bitwise() -> OpsBitwiseBaseline:
    rng = np.random.default_rng(4)
    a = rng.integers(0, 255, size=(2, 3), dtype=np.uint8)
    b = rng.integers(0, 255, size=(2, 3), dtype=np.uint8)
    return OpsBitwiseBaseline(
        bit_a=a,
        bit_b=b,
        and_out=np.bitwise_and(a, b),
        or_out=np.bitwise_or(a, b),
        xor_out=np.bitwise_xor(a, b),
        not_out=np.bitwise_not(a),
        shl_out=((a.astype(np.uint16) << 1) & 0xFF).astype(np.uint8),
        shr_out=(a >> 1).astype(np.uint8),
        popcount_out=popcount_u8(a),
    )


def build_ops_rounding() -> OpsRoundingBaseline:
    x = np.array([-2.3, -1.1, -0.2, 0.2, 1.1, 2.7], dtype=np.float32).reshape(2, 3)
    return OpsRoundingBaseline(
        round_x=x,
        floor_out=np.floor(x),
        ceil_out=np.ceil(x),
        round_out=np.round(x),
        trunc_out=np.trunc(x),
        sign_out=np.sign(x).astype(np.int8),
        recip_out=1.0 / x,
    )


def build_ops_reduce() -> OpsReduceBaseline:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    return OpsReduceBaseline(
        reduce_x=x,
        sum_axis_out=x.sum(axis=1, keepdims=True),
        mean_axis_out=x.mean(axis=1, keepdims=True),
        prod_axis_out=x.prod(axis=1, keepdims=True),
        min_axis_out=x.min(axis=1, keepdims=True),
        max_axis_out=x.max(axis=1, keepdims=True),
        argmax_axis_out=np.argmax(x, axis=1).astype(np.int64).reshape(2, 1),
        argmin_axis_out=np.argmin(x, axis=1).astype(np.int64).reshape(2, 1),
    )


def build_ops_cast() -> OpsCastBaseline:
    f32 = np.array([-5.0, 0.0, 3.2, 127.0, 200.0, 250.0], dtype=np.float32)
    i32 = np.array([-3, 0, 5, 11, -8, 42], dtype=np.int32)
    return OpsCastBaseline(
        cast_f32=f32,
        cast_f32_to_i32=f32.astype(np.int32),
        cast_f32_to_u8=f32.astype(np.uint8),
        cast_i32=i32,
        cast_i32_to_f32=i32.astype(np.float32),
    )


def build_ops_accumulate_inplace() -> OpsAccumulateInplaceBaseline:
    rng = np.random.default_rng(6)
    acc_a = rng.normal(size=(2, 3)).astype(np.float16)
    acc_b = rng.normal(size=(2, 3)).astype(np.float16)
    acc_out = acc_a.astype(np.float32) + acc_b.astype(np.float32)
    inplace_a = rng.normal(size=(2, 3)).astype(np.float32)
    inplace_b = rng.normal(size=(2, 3)).astype(np.float32)
    return OpsAccumulateInplaceBaseline(
        acc_a=acc_a,
        acc_b=acc_b,
        acc_out=acc_out,
        inplace_a=inplace_a,
        inplace_b=inplace_b,
        inplace_out=inplace_a + inplace_b,
    )


def build_ops_misc() -> OpsMiscBaseline:
    rng = np.random.default_rng(7)
    fma_a = rng.normal(size=(2, 3)).astype(np.float32)
    fma_b = rng.normal(size=(2, 3)).astype(np.float32)
    fma_c = rng.normal(size=(2, 3)).astype(np.float32)
    floor_div_a = rng.integers(1, 20, size=(2, 3), dtype=np.int32)
    floor_div_b = rng.integers(1, 5, size=(2, 3), dtype=np.int32)
    rem_a = rng.integers(1, 20, size=(2, 3), dtype=np.int32)
    rem_b = rng.integers(1, 6, size=(2, 3), dtype=np.int32)
    is_nan_x = np.array([np.nan, 0.0, 1.0, np.nan, -2.0, 3.0], dtype=np.float32).reshape(2, 3)
    is_inf_x = np.array([np.inf, -np.inf, 0.0, 1.0, -2.0, 3.0], dtype=np.float32).reshape(2, 3)
    is_neg_x = np.array([-1.0, 0.0, 1.0, -2.0, 3.0, -4.0], dtype=np.float32).reshape(2, 3)
    is_finite_x = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0, -2.0], dtype=np.float32).reshape(2, 3)
    return OpsMiscBaseline(
        fma_a=fma_a,
        fma_b=fma_b,
        fma_c=fma_c,
        fma_out=fma_a * fma_b + fma_c,
        floor_div_a=floor_div_a,
        floor_div_b=floor_div_b,
        floor_div_out=np.floor_divide(floor_div_a, floor_div_b),
        rem_a=rem_a,
        rem_b=rem_b,
        rem_out=np.remainder(rem_a, rem_b),
        is_nan_x=is_nan_x,
        is_nan_out=np.isnan(is_nan_x),
        is_inf_x=is_inf_x,
        is_inf_out=np.isinf(is_inf_x),
        is_neg_x=is_neg_x,
        is_neg_out=is_neg_x < 0,
        is_finite_x=is_finite_x,
        is_finite_out=np.array(np.isfinite(is_finite_x).all()),
    )


def quantize_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.vectorize(float_to_bf16_bits, otypes=[np.uint16])(values.astype(np.float32))
    return bf16_to_f32(bits)


def quantize_f8(values: np.ndarray) -> np.ndarray:
    bits = np.vectorize(float_to_f8_bits, otypes=[np.uint8])(values.astype(np.float32))
    return f8_to_f32(bits)


def build_ops_float_special() -> OpsFloatSpecialBaseline:
    rng = np.random.default_rng(8)
    base_a = rng.normal(size=(2, 3)).astype(np.float32)
    base_b = rng.normal(size=(2, 3)).astype(np.float32)
    base_relu = rng.normal(size=(2, 3)).astype(np.float32)

    f16_a = base_a.astype(np.float16)
    f16_b = base_b.astype(np.float16)
    f16_add = (f16_a + f16_b).astype(np.float16)
    f16_relu = np.maximum(base_relu.astype(np.float16), 0.0).astype(np.float16)

    bf16_a = quantize_bf16(base_a)
    bf16_b = quantize_bf16(base_b)
    bf16_add = quantize_bf16(bf16_a + bf16_b)
    bf16_relu = quantize_bf16(np.maximum(quantize_bf16(base_relu), 0.0))

    f8_a = quantize_f8(base_a)
    f8_b = quantize_f8(base_b)
    f8_add = quantize_f8(f8_a + f8_b)
    f8_relu = quantize_f8(np.maximum(quantize_f8(base_relu), 0.0))

    return OpsFloatSpecialBaseline(
        f16_a=TensorSpec(f16_a, dtype="f16"),
        f16_b=TensorSpec(f16_b, dtype="f16"),
        f16_add_out=TensorSpec(f16_add, dtype="f16"),
        f16_relu_x=TensorSpec(base_relu.astype(np.float16), dtype="f16"),
        f16_relu_out=TensorSpec(f16_relu, dtype="f16"),
        bf16_a=TensorSpec(bf16_a, dtype="bf16"),
        bf16_b=TensorSpec(bf16_b, dtype="bf16"),
        bf16_add_out=TensorSpec(bf16_add, dtype="bf16"),
        bf16_relu_x=TensorSpec(quantize_bf16(base_relu), dtype="bf16"),
        bf16_relu_out=TensorSpec(bf16_relu, dtype="bf16"),
        f8_a=TensorSpec(f8_a, dtype="f8"),
        f8_b=TensorSpec(f8_b, dtype="f8"),
        f8_add_out=TensorSpec(f8_add, dtype="f8"),
        f8_relu_x=TensorSpec(quantize_f8(base_relu), dtype="f8"),
        f8_relu_out=TensorSpec(f8_relu, dtype="f8"),
    )


def build_ops_packed() -> OpsPackedBaseline:
    i4_vals = np.array(
        [[-8, -3, 0, 7], [6, -1, -7, 3]],
        dtype=np.int8,
    )
    u2_vals = np.array(
        [[0, 1, 2, 3], [3, 2, 1, 0]],
        dtype=np.uint8,
    )
    i4_argmax = np.argmax(i4_vals, axis=1).astype(np.int64).reshape(2, 1)
    u2_argmin = np.argmin(u2_vals, axis=1).astype(np.int64).reshape(2, 1)
    return OpsPackedBaseline(
        packed_i4_x=TensorSpec(i4_vals, dtype="i4"),
        packed_i4_argmax_out=i4_argmax,
        packed_u2_x=TensorSpec(u2_vals, dtype="u2"),
        packed_u2_argmin_out=u2_argmin,
    )


def tensor_info(value: object) -> tuple[np.ndarray, str]:
    if isinstance(value, TensorSpec):
        arr = np.array(value.data)
        dtype = value.dtype if value.dtype is not None else str(arr.dtype)
        return arr, str(dtype)
    arr = np.array(value)
    return arr, str(arr.dtype)


def write_manifest(path: Path, tensors: dict[str, object]) -> None:
    manifest = {
        "file": path.name,
        "tensors": [
            {
                "name": name,
                "shape": list(tensor_info(value)[0].shape),
                "dtype": tensor_info(value)[1],
            }
            for name, value in tensors.items()
        ],
    }
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_baseline(data_dir: Path, name: str, baseline: object) -> None:
    path = data_dir / f"{name}.oinf"
    write_oinf(baseline, str(path))
    write_manifest(
        path,
        {field: getattr(baseline, field) for field in baseline.__dataclass_fields__},
    )
    print(f"Wrote {path}")


def _load_opspec() -> dict:
    with (ROOT / "ops.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_packed(dtype: str) -> bool:
    return dtype in {"i1", "i2", "i4", "u1", "u2", "u4", "t1", "t2"}


def _is_custom_float(dtype: str) -> bool:
    return dtype in {"bf16", "f8"}


def _is_bitset(dtype: str) -> bool:
    return dtype == "bitset"


def _is_float(dtype: str) -> bool:
    return dtype in {"f16", "f32", "f64", "bf16", "f8"}


def _dtype_numpy(dtype: str) -> np.dtype:
    return {
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "u64": np.uint64,
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
        "bool": np.bool_,
        "bitset": np.uint8,
    }[dtype]


def _dtype_range(dtype: str) -> tuple[int, int]:
    return {
        "i1": (-1, 0),
        "i2": (-2, 1),
        "i4": (-8, 7),
        "u1": (0, 1),
        "u2": (0, 3),
        "u4": (0, 15),
        "t1": (-1, 1),
        "t2": (-1, 1),
    }.get(dtype, (-3, 3))


def _quantize(dtype: str, values: np.ndarray) -> np.ndarray:
    if dtype == "bf16":
        return quantize_bf16(values.astype(np.float32))
    if dtype == "f8":
        return quantize_f8(values.astype(np.float32))
    if dtype == "f16":
        return values.astype(np.float16)
    return values


def _make_values(dtype: str, shape: tuple[int, ...], seed: int, op: str, role: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = int(np.prod(shape))
    if dtype == "bool":
        values = (rng.integers(0, 2, size=size) == 1).reshape(shape)
        return values.astype(np.bool_)
    if _is_float(dtype):
        values = rng.normal(size=size).astype(np.float32).reshape(shape)
        if op in {"div", "floor_div", "rem", "recip"} and role == "b":
            values = np.where(np.abs(values) < 0.5, 1.0, values)
        if op == "recip":
            values = np.where(np.abs(values) < 0.5, 1.0, values)
        return _quantize(dtype, values)
    if _is_bitset(dtype):
        values = rng.integers(0, 4, size=size, dtype=np.uint8).reshape(shape)
        return values
    if _is_packed(dtype):
        lo, hi = _dtype_range(dtype)
        if op in {"add", "sub", "mul", "fma", "div", "floor_div", "rem", "matmul"}:
            if lo < 0:
                lo = max(lo, -1 if op == "matmul" else -2)
                hi = min(hi, 1 if op == "matmul" else 2)
            else:
                lo = max(lo, 0)
                hi = min(hi, 1 if op == "matmul" else 2)
        values = rng.integers(lo, hi + 1, size=size, dtype=np.int8).reshape(shape)
        return values
    if dtype.startswith("u"):
        values = rng.integers(0, 4, size=size, dtype=_dtype_numpy(dtype)).reshape(shape)
        return values
    values = rng.integers(-3, 4, size=size, dtype=_dtype_numpy(dtype)).reshape(shape)
    if op in {"div", "floor_div", "rem"} and role == "b":
        values = np.abs(values)
        values = np.where(values == 0, 1, values)
        values = values.astype(_dtype_numpy(dtype))
    return values


def _tensor_spec(name: str, dtype: str, values: np.ndarray) -> tuple[str, object]:
    if _is_packed(dtype) or _is_custom_float(dtype) or dtype in {"f16", "bitset"}:
        return name, TensorSpec(values, dtype=dtype)
    return name, values.astype(_dtype_numpy(dtype))


def _default_attrs(op: str, dtype: str, out_dtype: str, mode: str) -> list[dict]:
    attrs: list[dict] = []
    if op in {"div", "floor_div", "recip"}:
        attrs.append({"name": "div_by_zero_mask", "kind": "scalar", "scalar_kind": "int", "value": 0})
    if op == "clamp":
        if _is_float(dtype):
            attrs.append({"name": "min", "kind": "scalar", "scalar_kind": "float", "value": 0.0})
            attrs.append({"name": "max", "kind": "scalar", "scalar_kind": "float", "value": 3.0})
        else:
            attrs.append({"name": "min", "kind": "scalar", "scalar_kind": "int", "value": 0})
            attrs.append({"name": "max", "kind": "scalar", "scalar_kind": "int", "value": 3})
    if op == "relu":
        if _is_float(dtype):
            attrs.append({"name": "alpha", "kind": "scalar", "scalar_kind": "float", "value": 0.2})
            attrs.append({"name": "clamp_max", "kind": "scalar", "scalar_kind": "float", "value": 6.0})
        else:
            attrs.append({"name": "alpha", "kind": "scalar", "scalar_kind": "int", "value": 1})
            attrs.append({"name": "clamp_max", "kind": "scalar", "scalar_kind": "int", "value": 6})
    if op in {"sum_axis", "mean_axis", "prod_axis", "min_axis", "max_axis"}:
        attrs.append({"name": "axes", "kind": "int_list", "value": [1]})
        attrs.append({"name": "keepdims", "kind": "scalar", "scalar_kind": "bool", "value": True})
    if op in {"argmax_axis", "argmin_axis"}:
        attrs.append({"name": "axis", "kind": "scalar", "scalar_kind": "int", "value": 1})
        attrs.append({"name": "keepdims", "kind": "scalar", "scalar_kind": "bool", "value": True})
        attrs.append({"name": "select_first", "kind": "scalar", "scalar_kind": "bool", "value": True})
    if op in {"shl", "shr"}:
        attrs.append({"name": "bits", "kind": "scalar", "scalar_kind": "int", "value": 1})
    if op == "fill":
        if _is_float(dtype):
            attrs.append({"name": "value", "kind": "scalar", "scalar_kind": "float", "value": 1.5})
        else:
            attrs.append({"name": "value", "kind": "scalar", "scalar_kind": "int", "value": 1})
    if op == "cast":
        attrs.append({"name": "to", "kind": "dtype", "value": out_dtype})
        attrs.append({"name": "rounding_mode", "kind": "string", "value": "trunc"})
        attrs.append({"name": "saturate", "kind": "scalar", "scalar_kind": "bool", "value": True})
    if mode == "accumulate":
        attrs.append({"name": "acc", "kind": "dtype", "value": out_dtype})
    return attrs


def _apply_attrs(attrs: list[dict], name: str, default: object) -> object:
    for item in attrs:
        if item["name"] == name:
            return item["value"]
    return default


def _compute_op(op: str, inputs: list[np.ndarray], attrs: list[dict], dtype: str, out_dtype: str) -> np.ndarray:
    if op == "add":
        return inputs[0] + inputs[1]
    if op == "sub":
        return inputs[0] - inputs[1]
    if op == "mul":
        return inputs[0] * inputs[1]
    if op == "div":
        return inputs[0] / inputs[1]
    if op == "floor_div":
        if _is_float(dtype):
            return np.floor_divide(inputs[0], inputs[1])
        mask = _apply_attrs(attrs, "div_by_zero_mask", 0)
        lhs = inputs[0].astype(np.int64, copy=False)
        rhs = inputs[1].astype(np.int64, copy=False)
        out = np.zeros_like(lhs)
        zero = rhs == 0
        out[zero] = mask
        nz = ~zero
        if np.any(nz):
            lhs_nz = lhs[nz]
            rhs_nz = rhs[nz]
            q = np.trunc(lhs_nz / rhs_nz).astype(np.int64)
            r = lhs_nz - q * rhs_nz
            adjust = r < 0
            q = np.where(adjust & (rhs_nz > 0), q - 1, q)
            q = np.where(adjust & (rhs_nz < 0), q + 1, q)
            out[nz] = q
        return out
    if op == "rem":
        if _is_float(dtype):
            return np.remainder(inputs[0], inputs[1])
        lhs = inputs[0].astype(np.int64, copy=False)
        rhs = inputs[1].astype(np.int64, copy=False)
        out = np.zeros_like(lhs)
        zero = rhs == 0
        out[zero] = 0
        nz = ~zero
        if np.any(nz):
            lhs_nz = lhs[nz]
            rhs_nz = rhs[nz]
            q = np.trunc(lhs_nz / rhs_nz).astype(np.int64)
            out[nz] = lhs_nz - q * rhs_nz
        return out
    if op == "fma":
        return inputs[0] * inputs[1] + inputs[2]
    if op == "abs":
        return np.abs(inputs[0])
    if op == "neg":
        return -inputs[0]
    if op == "sign":
        return np.sign(inputs[0])
    if op == "recip":
        return 1.0 / inputs[0]
    if op == "min":
        return np.minimum(inputs[0], inputs[1])
    if op == "max":
        return np.maximum(inputs[0], inputs[1])
    if op == "clamp":
        min_val = _apply_attrs(attrs, "min", 0.0)
        max_val = _apply_attrs(attrs, "max", 1.0)
        return np.clip(inputs[0], min_val, max_val)
    if op == "floor":
        return np.floor(inputs[0])
    if op == "ceil":
        return np.ceil(inputs[0])
    if op == "round":
        return np.round(inputs[0])
    if op == "trunc":
        return np.trunc(inputs[0])
    if op == "and":
        return np.bitwise_and(inputs[0], inputs[1])
    if op == "or":
        return np.bitwise_or(inputs[0], inputs[1])
    if op == "xor":
        return np.bitwise_xor(inputs[0], inputs[1])
    if op == "not":
        return np.bitwise_not(inputs[0])
    if op == "shl":
        return np.left_shift(inputs[0], 1)
    if op == "shr":
        return np.right_shift(inputs[0], 1)
    if op == "popcount":
        values = inputs[0].astype(np.uint64)
        flat = values.reshape(-1, 1)
        counts = np.unpackbits(flat.view(np.uint8), axis=1).sum(axis=1)
        return counts.astype(np.uint8).reshape(values.shape)
    if op == "eq":
        return inputs[0] == inputs[1]
    if op == "ne":
        return inputs[0] != inputs[1]
    if op == "lt":
        return inputs[0] < inputs[1]
    if op == "le":
        return inputs[0] <= inputs[1]
    if op == "gt":
        return inputs[0] > inputs[1]
    if op == "ge":
        return inputs[0] >= inputs[1]
    if op == "filter":
        cond = inputs[2] != 0
        return np.where(cond, inputs[0], inputs[1])
    if op == "is_nan":
        return np.isnan(inputs[0])
    if op == "is_inf":
        return np.isinf(inputs[0])
    if op == "is_neg":
        return inputs[0] < 0
    if op == "is_finite":
        return np.array(np.isfinite(inputs[0]).all())
    if op == "sum_axis":
        return inputs[0].sum(axis=1, keepdims=True)
    if op == "mean_axis":
        return inputs[0].mean(axis=1, keepdims=True)
    if op == "prod_axis":
        return inputs[0].prod(axis=1, keepdims=True)
    if op == "min_axis":
        return inputs[0].min(axis=1, keepdims=True)
    if op == "max_axis":
        return inputs[0].max(axis=1, keepdims=True)
    if op == "argmax_axis":
        return np.argmax(inputs[0], axis=1).astype(np.int64).reshape(-1, 1)
    if op == "argmin_axis":
        return np.argmin(inputs[0], axis=1).astype(np.int64).reshape(-1, 1)
    if op == "cast":
        target = out_dtype
        values = inputs[0]
        if target in {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}:
            if _is_float(dtype):
                values = np.trunc(values)
            if target.startswith("u"):
                min_val = 0
                max_val = np.iinfo(_dtype_numpy(target)).max
            else:
                min_val = np.iinfo(_dtype_numpy(target)).min
                max_val = np.iinfo(_dtype_numpy(target)).max
            values = np.clip(values, min_val, max_val)
            return values.astype(_dtype_numpy(target))
        if target == "f16":
            return values.astype(np.float16)
        if target == "bf16":
            return quantize_bf16(values.astype(np.float32))
        if target == "f8":
            return quantize_f8(values.astype(np.float32))
        if target == "f32":
            return values.astype(np.float32)
        if target == "f64":
            return values.astype(np.float64)
    if op == "fill":
        value = _apply_attrs(attrs, "value", 0)
        return np.full_like(inputs[0], value)
    if op == "relu":
        alpha = float(_apply_attrs(attrs, "alpha", 0.0))
        clamp_max = float(_apply_attrs(attrs, "clamp_max", 6.0))
        out = np.where(inputs[0] >= 0, inputs[0], inputs[0] * alpha)
        return np.minimum(out, clamp_max)
    if op == "matmul":
        return np.matmul(inputs[0], inputs[1])
    raise ValueError(f"Unsupported op {op}")


def _cast_output(dtype: str, output: np.ndarray) -> np.ndarray:
    if dtype == "bool":
        return output.astype(np.bool_)
    if _is_bitset(dtype):
        return output.astype(np.uint8)
    if _is_packed(dtype):
        return _wrap_packed(dtype, output)
    if _is_float(dtype):
        return _quantize(dtype, output)
    return output.astype(_dtype_numpy(dtype))


def _wrap_packed(dtype: str, output: np.ndarray) -> np.ndarray:
    values = output.astype(np.int64, copy=False)
    if dtype in {"u1", "u2", "u4"}:
        bits = {"u1": 1, "u2": 2, "u4": 4}[dtype]
        mask = (1 << bits) - 1
        return (values & mask).astype(np.uint8)
    if dtype == "t1":
        return np.where(values < 0, -1, 1).astype(np.int8)
    bits = {"i1": 1, "i2": 2, "i4": 4, "t2": 2}[dtype]
    mask = (1 << bits) - 1
    masked = values & mask
    sign_bit = 1 << (bits - 1)
    wrapped = np.where((masked & sign_bit) != 0, masked - (1 << bits), masked)
    return wrapped.astype(np.int8)


def generate_ops_full_matrix() -> None:
    opspec = _load_opspec()
    dtype_sets = opspec["dtype_sets"]
    ops = opspec["ops"]

    out_dir = ROOT / "tests/openinfer/ops/baseline/data/full_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    cases: list[dict] = []

    inplace_skip_ops = {"sign"}
    for op_entry in ops:
        op = op_entry["kind"]
        input_arity = op_entry["inputs"]["arity"]
        if input_arity != "fixed":
            continue
        input_count = op_entry["inputs"]["count"]
        dtype_support = dtype_sets[op_entry["dtype_support_ref"]]
        normal_dtypes = [
            dtype for dtype in dtype_support.get("normal", []) if dtype != "bitset"
        ]
        accumulate_pairs = [
            pair
            for pair in dtype_support.get("accumulate", [])
            if pair.get("input") != "bitset" and pair.get("acc") != "bitset"
        ]
        type_rule = op_entry["type_rule"]

        tensor_map: dict[str, object] = {}
        output_dtypes_ref = op_entry.get("output_dtypes_ref")
        output_dtypes = opspec.get("output_dtypes", {})
        for dtype in normal_dtypes:
            base_shape = (2, 3)
            if op == "matmul":
                shapes = [(2, 3), (3, 4)]
            else:
                shapes = [base_shape for _ in range(input_count)]

            inputs = []
            input_names = []
            for idx in range(input_count):
                values = _make_values(dtype, shapes[idx], seed=idx + 10, op=op, role="a" if idx == 0 else "b")
                name = f"{op}_normal_{dtype}_in{idx}"
                key, spec = _tensor_spec(name, dtype, values)
                tensor_map[key] = spec
                input_names.append(name)
                inputs.append(values.astype(np.float32) if _is_float(dtype) else values)

            if type_rule["kind"] == "fixed":
                out_candidates = [type_rule["dtype"]]
            elif type_rule["kind"] == "acc_from_attr" and output_dtypes_ref:
                out_candidates = output_dtypes.get(output_dtypes_ref, [])
            else:
                out_candidates = [dtype]

            for out_dtype in out_candidates:
                attrs = _default_attrs(op, dtype, out_dtype, "normal")
                with np.errstate(divide="ignore", invalid="ignore"):
                    output = _compute_op(op, inputs, attrs, dtype, out_dtype)
                    output = _cast_output(out_dtype, output)
                out_name = f"{op}_normal_{dtype}_to_{out_dtype}_out"
                key, spec = _tensor_spec(out_name, out_dtype, output)
                tensor_map[key] = spec
                cases.append({
                    "file": f"full_matrix/{op}.oinf",
                    "name": f"{op}_normal_{dtype}_to_{out_dtype}",
                    "op": op,
                    "mode": "normal",
                    "input_dtype": dtype,
                    "output_dtype": out_dtype,
                    "inputs": input_names,
                    "output_var": out_name,
                    "expected": out_name,
                    "attrs": attrs,
                })

                if op_entry["inplace"] == "allow" and out_dtype == dtype and op not in inplace_skip_ops:
                    inplace_inputs = inputs
                    inplace_input_names = input_names
                    if op == "matmul":
                        inplace_shapes = [(2, 3), (3, 3)]
                        inplace_inputs = []
                        inplace_input_names = []
                        for idx in range(input_count):
                            values = _make_values(
                                dtype,
                                inplace_shapes[idx],
                                seed=idx + 20,
                                op=op,
                                role="a" if idx == 0 else "b",
                            )
                            name = f"{op}_inplace_{dtype}_in{idx}"
                            key, spec = _tensor_spec(name, dtype, values)
                            tensor_map[key] = spec
                            inplace_input_names.append(name)
                            inplace_inputs.append(
                                values.astype(np.float32) if _is_float(dtype) else values
                            )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        inplace_out = _compute_op(op, inplace_inputs, attrs, dtype, out_dtype)
                        inplace_out = _cast_output(out_dtype, inplace_out)
                    inplace_name = f"{op}_inplace_{dtype}_to_{out_dtype}_out"
                    key, spec = _tensor_spec(inplace_name, out_dtype, inplace_out)
                    tensor_map[key] = spec
                    cases.append({
                        "file": f"full_matrix/{op}.oinf",
                        "name": f"{op}_inplace_{dtype}_to_{out_dtype}",
                        "op": op,
                        "mode": "inplace",
                        "input_dtype": dtype,
                        "output_dtype": out_dtype,
                        "inputs": inplace_input_names,
                        "output_var": inplace_input_names[0],
                        "expected": inplace_name,
                        "attrs": attrs,
                    })

        for pair in accumulate_pairs:
            in_dtype = pair["input"]
            acc_dtype = pair["acc"]
            base_shape = (2, 3)
            if op == "matmul":
                shapes = [(2, 3), (3, 4)]
            else:
                shapes = [base_shape for _ in range(input_count)]
            inputs = []
            input_names = []
            for idx in range(input_count):
                values = _make_values(in_dtype, shapes[idx], seed=idx + 30, op=op, role="a" if idx == 0 else "b")
                name = f"{op}_acc_{in_dtype}_{acc_dtype}_in{idx}"
                key, spec = _tensor_spec(name, in_dtype, values)
                tensor_map[key] = spec
                input_names.append(name)
                inputs.append(values.astype(np.float32) if _is_float(in_dtype) else values)

            attrs = _default_attrs(op, in_dtype, acc_dtype, "accumulate")
            with np.errstate(divide="ignore", invalid="ignore"):
                output = _compute_op(op, inputs, attrs, in_dtype, acc_dtype)
                output = _cast_output(acc_dtype, output)
            out_name = f"{op}_acc_{in_dtype}_{acc_dtype}_out"
            key, spec = _tensor_spec(out_name, acc_dtype, output)
            tensor_map[key] = spec
            cases.append({
                "file": f"full_matrix/{op}.oinf",
                "name": f"{op}_acc_{in_dtype}_{acc_dtype}",
                "op": op,
                "mode": "accumulate",
                "input_dtype": in_dtype,
                "output_dtype": acc_dtype,
                "acc_dtype": acc_dtype,
                "inputs": input_names,
                "output_var": out_name,
                "expected": out_name,
                "attrs": attrs,
            })

        if tensor_map:
            Baseline = make_dataclass(
                f"{op}_baseline",
                [(name, object) for name in tensor_map.keys()],
            )
            instance = Baseline(**tensor_map)
            path = out_dir / f"{op}.oinf"
            write_oinf(instance, str(path))

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"cases": cases}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    inventory = []
    for op_entry in ops:
        support = dtype_sets[op_entry["dtype_support_ref"]]
        skipped = []
        if "bitset" in support.get("normal", []):
            skipped.append("bitset")
        for pair in support.get("accumulate", []):
            if pair.get("input") == "bitset" or pair.get("acc") == "bitset":
                skipped.append("bitset")
        inventory.append(
            {
                "op": op_entry["kind"],
                "normal": support.get("normal", []),
                "accumulate": support.get("accumulate", []),
                "inplace": op_entry["inplace"],
                "type_rule": op_entry["type_rule"],
                "output_dtypes_ref": op_entry.get("output_dtypes_ref"),
                "skipped_dtypes": sorted(set(skipped)),
            }
        )
    with (out_dir / "inventory.json").open("w", encoding="utf-8") as handle:
        json.dump({"ops": inventory}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    data_dir = ROOT / "tests/openinfer/ops/baseline/data"
    data_dir.mkdir(parents=True, exist_ok=True)

    write_baseline(data_dir, "ops_basic", build_ops_basic())
    write_baseline(data_dir, "ops_matmul", build_ops_matmul())
    write_baseline(data_dir, "ops_broadcast", build_ops_broadcast())
    write_baseline(data_dir, "ops_compare", build_ops_compare())
    write_baseline(data_dir, "ops_bitwise", build_ops_bitwise())
    write_baseline(data_dir, "ops_rounding", build_ops_rounding())
    write_baseline(data_dir, "ops_reduce", build_ops_reduce())
    write_baseline(data_dir, "ops_cast", build_ops_cast())
    write_baseline(data_dir, "ops_accumulate_inplace", build_ops_accumulate_inplace())
    write_baseline(data_dir, "ops_misc", build_ops_misc())
    write_baseline(data_dir, "ops_float_special", build_ops_float_special())
    write_baseline(data_dir, "ops_packed", build_ops_packed())
    generate_ops_full_matrix()


if __name__ == "__main__":
    main()
