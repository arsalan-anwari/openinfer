#!/usr/bin/env python3
"""
Generate Vulkan per-dtype Slang shaders for ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VULKAN_DIR = ROOT / "openinfer" / "src" / "ops" / "vulkan"


@dataclass(frozen=True)
class DTypeSpec:
    name: str
    kind: str  # "float", "signed", "unsigned", "bool", "bitset", "packed_signed", "packed_unsigned"
    bits: int = 0


FLOAT_DTYPES = [DTypeSpec("f32", "float"), DTypeSpec("f64", "float")]
SIGNED_DTYPES = [
    DTypeSpec("i8", "signed"),
    DTypeSpec("i16", "signed"),
    DTypeSpec("i32", "signed"),
    DTypeSpec("i64", "signed"),
]
UNSIGNED_DTYPES = [
    DTypeSpec("u8", "unsigned"),
    DTypeSpec("u16", "unsigned"),
    DTypeSpec("u32", "unsigned"),
    DTypeSpec("u64", "unsigned"),
]
PACKED_SIGNED = [
    DTypeSpec("i4", "packed_signed", 4),
    DTypeSpec("i2", "packed_signed", 2),
    DTypeSpec("i1", "packed_signed", 1),
]
PACKED_UNSIGNED = [
    DTypeSpec("u4", "packed_unsigned", 4),
    DTypeSpec("u2", "packed_unsigned", 2),
    DTypeSpec("u1", "packed_unsigned", 1),
]
BOOL_DTYPE = [DTypeSpec("bool", "bool")]
BITSET_DTYPE = [DTypeSpec("bitset", "bitset")]

BASE_ADD_MUL_FILL = (
    FLOAT_DTYPES
    + SIGNED_DTYPES
    + UNSIGNED_DTYPES
    + BOOL_DTYPE
    + BITSET_DTYPE
    + PACKED_SIGNED
    + PACKED_UNSIGNED
)
BASE_ABS = FLOAT_DTYPES + SIGNED_DTYPES + PACKED_SIGNED
BASE_RELU = [DTypeSpec("i4", "packed_signed", 4)] + SIGNED_DTYPES + FLOAT_DTYPES
BASE_IS_FINITE = FLOAT_DTYPES

ACC_SIGNED = [
    ("i8", "i16"),
    ("i8", "i32"),
    ("i8", "i64"),
    ("i16", "i32"),
    ("i16", "i64"),
    ("i32", "i64"),
]
ACC_UNSIGNED = [
    ("u8", "u16"),
    ("u8", "u32"),
    ("u8", "u64"),
    ("u16", "u32"),
    ("u16", "u64"),
    ("u32", "u64"),
]
ACC_PACKED_SIGNED = [
    ("i4", "i8"),
    ("i4", "i16"),
    ("i4", "i32"),
    ("i4", "i64"),
    ("i2", "i8"),
    ("i2", "i16"),
    ("i2", "i32"),
    ("i2", "i64"),
    ("i1", "i8"),
    ("i1", "i16"),
    ("i1", "i32"),
    ("i1", "i64"),
]
ACC_PACKED_UNSIGNED = [
    ("u4", "u8"),
    ("u4", "u16"),
    ("u4", "u32"),
    ("u4", "u64"),
    ("u2", "u8"),
    ("u2", "u16"),
    ("u2", "u32"),
    ("u2", "u64"),
    ("u1", "u8"),
    ("u1", "u16"),
    ("u1", "u32"),
    ("u1", "u64"),
]


def matmul_base(op: str, dtype: DTypeSpec, inplace: bool) -> str:
    header = """struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;
"""
    name = f"{op}_inplace_{dtype.name}" if inplace else f"{op}_{dtype.name}"
    if dtype.kind in ("packed_signed", "packed_unsigned"):
        signed = dtype.kind == "packed_signed"
        read_fn = "packed_read_signed" if signed else "packed_read_unsigned"
        cast = "int" if signed else "uint"
        buffers = (
            "struct MatmulPackedBuffers {\n"
            "    [[vk::binding(0, 0)]] ByteAddressBuffer input0;\n"
            "    [[vk::binding(1, 0)]] ByteAddressBuffer input1;\n"
            "    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;\n"
            "};\n\n"
            "ParameterBlock<MatmulPackedBuffers> gMatmulPacked;\n"
        )
        body = f"""uint idx = tid.x;
if (idx < pc.len) {{
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    {cast} acc = 0;
    for (uint kk = 0; kk < pc.k; ++kk) {{
        {cast} a = {read_fn}(gMatmulPacked.input0, a_row + kk, {dtype.bits});
        {cast} b = {read_fn}(gMatmulPacked.input1, kk * pc.n + col, {dtype.bits});
        acc = acc + (a * b);
    }}
    packed_write(gMatmulPacked.output0, idx, {dtype.bits}, (uint)acc);
}}
"""
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    if dtype.kind == "bool":
        buffers = (
            "struct MatmulBuffers<T> {\n"
            "    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;\n"
            "    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;\n"
            "    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;\n"
            "};\n\n"
            "ParameterBlock<MatmulBuffers<uint>> gMatmul_uint;\n"
        )
        body = """uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = 0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint a = gMatmul_uint.input0[a_row + kk];
        uint b = gMatmul_uint.input1[kk * pc.n + col];
        if (a != 0 && b != 0) {
            acc = 1;
            break;
        }
    }
    gMatmul_uint.output0[idx] = acc;
}
"""
        return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    if dtype.kind == "bitset":
        buffers = (
            "struct MatmulBuffers<T> {\n"
            "    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;\n"
            "    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;\n"
            "    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;\n"
            "};\n\n"
            "ParameterBlock<MatmulBuffers<uint>> gMatmul_uint;\n"
        )
        body = """uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = 0u;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint a = gMatmul_uint.input0[a_row + kk] & 0xFFu;
        uint b = gMatmul_uint.input1[kk * pc.n + col] & 0xFFu;
        acc = (acc + (a * b)) & 0xFFu;
    }
    gMatmul_uint.output0[idx] = acc;
}
"""
        return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = (
        "struct MatmulBuffers<T> {\n"
        "    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;\n"
        "    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;\n"
        "    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;\n"
        "};\n\n"
        f"ParameterBlock<MatmulBuffers<{dtype_to_ctype(dtype)}>> gMatmul_{dtype_to_ctype(dtype)};\n"
    )
    body = f"""uint idx = tid.x;
if (idx < pc.len) {{
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    {dtype_to_ctype(dtype)} acc = ({dtype_to_ctype(dtype)})0;
    for (uint kk = 0; kk < pc.k; ++kk) {{
        acc = acc + (gMatmul_{dtype_to_ctype(dtype)}.input0[a_row + kk]
            * gMatmul_{dtype_to_ctype(dtype)}.input1[kk * pc.n + col]);
    }}
    gMatmul_{dtype_to_ctype(dtype)}.output0[idx] = acc;
}}
"""
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def matmul_accumulate(in_name: str, out_name: str, signed: bool) -> str:
    header = """struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;
"""
    name = f"matmul_{in_name}_{out_name}"
    in_kind, in_bits = packed_kind(in_name)
    out_type = acc_ctype(out_name)
    if in_kind:
        read_fn = "packed_read_signed" if in_kind == "signed" else "packed_read_unsigned"
        cast = "int" if in_kind == "signed" else "uint"
        buffers = (
            "struct MatmulAccBuffers {\n"
            "    [[vk::binding(0, 0)]] ByteAddressBuffer input0;\n"
            "    [[vk::binding(1, 0)]] ByteAddressBuffer input1;\n"
            f"    [[vk::binding(2, 0)]] RWStructuredBuffer<{out_type}, Std430DataLayout> output0;\n"
            "};\n\n"
            "ParameterBlock<MatmulAccBuffers> gMatmulAcc;\n"
        )
        body = f"""uint idx = tid.x;
if (idx < pc.len) {{
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    {out_type} acc = ({out_type})0;
    for (uint kk = 0; kk < pc.k; ++kk) {{
        {cast} a = {read_fn}(gMatmulAcc.input0, a_row + kk, {in_bits});
        {cast} b = {read_fn}(gMatmulAcc.input1, kk * pc.n + col, {in_bits});
        acc = acc + ({out_type})({cast})(a * b);
    }}
    gMatmulAcc.output0[idx] = acc;
}}
"""
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = (
        "struct MatmulAccBuffers {\n"
        f"    [[vk::binding(0, 0)]] StructuredBuffer<{acc_ctype(in_name)}, Std430DataLayout> input0;\n"
        f"    [[vk::binding(1, 0)]] StructuredBuffer<{acc_ctype(in_name)}, Std430DataLayout> input1;\n"
        f"    [[vk::binding(2, 0)]] RWStructuredBuffer<{out_type}, Std430DataLayout> output0;\n"
        "};\n\n"
        "ParameterBlock<MatmulAccBuffers> gMatmulAcc;\n"
    )
    body = f"""uint idx = tid.x;
if (idx < pc.len) {{
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    {out_type} acc = ({out_type})0;
    for (uint kk = 0; kk < pc.k; ++kk) {{
        {out_type} a = ({out_type})gMatmulAcc.input0[a_row + kk];
        {out_type} b = ({out_type})gMatmulAcc.input1[kk * pc.n + col];
        acc = acc + (a * b);
    }}
    gMatmulAcc.output0[idx] = acc;
}}
"""
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def header_push(len_only: bool = True) -> str:
    if len_only:
        return """struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;
"""
    return """struct PushConsts {
    uint len;
    uint value_bits;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;
"""


def include_packed() -> str:
    return '#include "../../../packed/packed_utils.slang"\n'


def scalar_buffers(op: str, ctype: str, inplace: bool, include_input1: bool = True) -> str:
    out = []
    out.append(f"struct {op.capitalize()}Buffers<T> {{")
    out.append("    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;")
    if include_input1:
        out.append("    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;")
    out.append("    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;")
    out.append("};\n")
    out.append(f"ParameterBlock<{op.capitalize()}Buffers<{ctype}>> g{op.capitalize()}_{ctype};\n")
    return "\n".join(out)


def packed_buffers(op: str, inplace: bool, include_input1: bool = True) -> str:
    out = []
    out.append(f"struct {op.capitalize()}PackedBuffers {{")
    out.append("    [[vk::binding(0, 0)]] ByteAddressBuffer input0;")
    if include_input1:
        out.append("    [[vk::binding(1, 0)]] ByteAddressBuffer input1;")
    out.append("    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;")
    out.append("};\n")
    out.append(f"ParameterBlock<{op.capitalize()}PackedBuffers> g{op.capitalize()}Packed;\n")
    return "\n".join(out)


def add_mul_body(op: str, dtype: DTypeSpec, inplace: bool) -> str:
    if dtype.kind == "bool":
        if op == "add":
            return f"""uint i = tid.x;
if (i < pc.len) {{
    uint a = g{op.capitalize()}_uint.input0[i];
    uint b = g{op.capitalize()}_uint.input1[i];
    g{op.capitalize()}_uint.output0[i] = (a != 0u || b != 0u) ? 1u : 0u;
}}
"""
        return f"""uint i = tid.x;
if (i < pc.len) {{
    uint a = g{op.capitalize()}_uint.input0[i];
    uint b = g{op.capitalize()}_uint.input1[i];
    g{op.capitalize()}_uint.output0[i] = (a != 0u && b != 0u) ? 1u : 0u;
}}
"""
    if dtype.kind == "bitset":
        op_expr = "+" if op == "add" else "*"
        return f"""uint i = tid.x;
if (i < pc.len) {{
    uint a = g{op.capitalize()}_uint.input0[i] & 0xFFu;
    uint b = g{op.capitalize()}_uint.input1[i] & 0xFFu;
    g{op.capitalize()}_uint.output0[i] = (a {op_expr} b) & 0xFFu;
}}
"""
    if dtype.kind in ("packed_signed", "packed_unsigned"):
        op_expr = "+" if op == "add" else "*"
        signed = dtype.kind == "packed_signed"
        read_fn = "packed_read_signed" if signed else "packed_read_unsigned"
        cast = "int" if signed else "uint"
        return f"""uint i = tid.x;
if (i < pc.len) {{
    {cast} a = {read_fn}(g{op.capitalize()}Packed.input0, i, {dtype.bits});
    {cast} b = {read_fn}(g{op.capitalize()}Packed.input1, i, {dtype.bits});
    {cast} y = a {op_expr} b;
    packed_write(g{op.capitalize()}Packed.output0, i, {dtype.bits}, (uint)y);
}}
"""
    op_expr = "+" if op == "add" else "*"
    return f"""uint i = tid.x;
if (i < pc.len) {{
    g{op.capitalize()}_{dtype_to_ctype(dtype)}.output0[i] =
        g{op.capitalize()}_{dtype_to_ctype(dtype)}.input0[i] {op_expr} g{op.capitalize()}_{dtype_to_ctype(dtype)}.input1[i];
}}
"""


def abs_body(dtype: DTypeSpec) -> str:
    if dtype.kind in ("packed_signed",):
        return f"""uint i = tid.x;
if (i < pc.len) {{
    int x = packed_read_signed(gAbsPacked.input0, i, {dtype.bits});
    int y = x < 0 ? -x : x;
    packed_write(gAbsPacked.output0, i, {dtype.bits}, (uint)y);
}}
"""
    if dtype.kind == "float":
        return f"""uint i = tid.x;
if (i < pc.len) {{
    {dtype_to_ctype(dtype)} x = gAbs_{dtype_to_ctype(dtype)}.input0[i];
    gAbs_{dtype_to_ctype(dtype)}.output0[i] = abs(x);
}}
"""
    return f"""uint i = tid.x;
if (i < pc.len) {{
    int x = gAbs_{dtype_to_ctype(dtype)}.input0[i];
    gAbs_{dtype_to_ctype(dtype)}.output0[i] = x < 0 ? -x : x;
}}
"""


def fill_body(dtype: DTypeSpec) -> str:
    if dtype.kind == "float":
        return f"""uint i = tid.x;
if (i < pc.len) {{
    float value = asfloat(pc.value_bits);
    gFill_{dtype_to_ctype(dtype)}.output0[i] = ({dtype_to_ctype(dtype)})value;
}}
"""
    if dtype.kind == "bool":
        return """uint i = tid.x;
if (i < pc.len) {
    float value = asfloat(pc.value_bits);
    gFill_uint.output0[i] = value != 0.0f ? 1u : 0u;
}
"""
    if dtype.kind == "bitset":
        return """uint i = tid.x;
if (i < pc.len) {
    float value = asfloat(pc.value_bits);
    uint v = ((uint)value) & 0xFFu;
    gFill_uint.output0[i] = v;
}
"""
    if dtype.kind in ("packed_signed", "packed_unsigned"):
        return f"""uint i = tid.x;
if (i < pc.len) {{
    float value = asfloat(pc.value_bits);
    uint raw = (uint)((int)value);
    packed_write(gFillPacked.output0, i, {dtype.bits}, raw);
}}
"""
    return f"""uint i = tid.x;
if (i < pc.len) {{
    float value = asfloat(pc.value_bits);
    gFill_{dtype_to_ctype(dtype)}.output0[i] = ({dtype_to_ctype(dtype)})value;
}}
"""


def relu_body(dtype: DTypeSpec) -> str:
    if dtype.kind == "float":
        return f"""uint i = tid.x;
if (i < pc.len) {{
    {dtype_to_ctype(dtype)} x = gRelu_{dtype_to_ctype(dtype)}.input0[i];
    float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
    gRelu_{dtype_to_ctype(dtype)}.output0[i] = ({dtype_to_ctype(dtype)})y;
}}
"""
    if dtype.kind in ("packed_signed",):
        return f"""uint i = tid.x;
if (i < pc.len) {{
    int x = packed_read_signed(gReluPacked.input0, i, {dtype.bits});
    float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
    if (y < -128.0f) {{
        y = -128.0f;
    }}
    if (y > 127.0f) {{
        y = 127.0f;
    }}
    int yi = (int)y;
    packed_write(gReluPacked.output0, i, {dtype.bits}, (uint)yi);
}}
"""
    min_val, max_val = relu_min_max(dtype.name)
    return f"""uint i = tid.x;
if (i < pc.len) {{
    {dtype_to_ctype(dtype)} x = gRelu_{dtype_to_ctype(dtype)}.input0[i];
    float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
    if (y < {min_val}) {{
        y = {min_val};
    }}
    if (y > {max_val}) {{
        y = {max_val};
    }}
    gRelu_{dtype_to_ctype(dtype)}.output0[i] = ({dtype_to_ctype(dtype)})y;
}}
"""


def relu_min_max(dtype_name: str) -> tuple[str, str]:
    if dtype_name == "i8":
        return ("-128.0f", "127.0f")
    if dtype_name == "i16":
        return ("-32768.0f", "32767.0f")
    if dtype_name == "i32":
        return ("-2147483648.0f", "2147483647.0f")
    if dtype_name == "i64":
        return ("-9223372036854775808.0f", "9223372036854775807.0f")
    return ("-128.0f", "127.0f")


def is_finite_body(dtype: DTypeSpec) -> str:
    limit = "3.402823466e38f" if dtype.name == "f32" else "1.7976931348623157e308"
    return f"""uint ok = 1u;
for (uint i = 0; i < pc.len; ++i) {{
    {dtype_to_ctype(dtype)} x = gIsFinite_{dtype_to_ctype(dtype)}.input0[i];
    {dtype_to_ctype(dtype)} abs_x = abs(x);
    if (x != x || abs_x > ({dtype_to_ctype(dtype)}){limit}) {{
        ok = 0u;
        break;
    }}
}}
gIsFinite_{dtype_to_ctype(dtype)}.output0[0] = ok;
"""


def dtype_to_ctype(dtype: DTypeSpec) -> str:
    if dtype.kind in ("float",):
        return "float" if dtype.name != "f64" else "double"
    if dtype.kind in ("signed",):
        return "int64_t" if dtype.name == "i64" else "int"
    if dtype.kind in ("unsigned", "bool", "bitset"):
        return "uint64_t" if dtype.name == "u64" else "uint"
    return "uint"


def make_add_mul(op: str, inplace: bool, dtype: DTypeSpec) -> str:
    name = f"{op}_inplace_{dtype.name}" if inplace else f"{op}_{dtype.name}"
    header = header_push()
    if dtype.kind in ("packed_signed", "packed_unsigned"):
        buffers = packed_buffers(op, inplace, True)
        body = add_mul_body(op, dtype, inplace)
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = scalar_buffers(op, dtype_to_ctype(dtype), inplace, True)
    body = add_mul_body(op, dtype, inplace)
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def make_abs(inplace: bool, dtype: DTypeSpec) -> str:
    name = f"abs_inplace_{dtype.name}" if inplace else f"abs_{dtype.name}"
    header = header_push()
    if dtype.kind in ("packed_signed",):
        buffers = packed_buffers("abs", inplace, False)
        body = abs_body(dtype)
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = scalar_buffers("abs", dtype_to_ctype(dtype), inplace, False)
    body = abs_body(dtype)
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def make_fill(inplace: bool, dtype: DTypeSpec) -> str:
    name = f"fill_inplace_{dtype.name}" if inplace else f"fill_{dtype.name}"
    header = header_push(len_only=False)
    if dtype.kind in ("packed_signed", "packed_unsigned"):
        buffers = packed_buffers("fill", inplace, False)
        body = fill_body(dtype)
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = scalar_buffers("fill", dtype_to_ctype(dtype), inplace, False)
    body = fill_body(dtype)
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def make_relu(dtype: DTypeSpec) -> str:
    name = f"relu_{dtype.name}"
    header = """struct PushConsts {
    uint len;
    uint neg_bits;
    uint clamp_bits;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;
"""
    relu_fn = """float relu_float(float x, float negative_slope, float clamp_max) {
    float y = x >= 0.0f ? x : x * negative_slope;
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}
"""
    if dtype.kind in ("packed_signed",):
        buffers = packed_buffers("relu", False, False)
        body = relu_body(dtype)
        return f"""{include_packed()}{header}
{relu_fn}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = scalar_buffers("relu", dtype_to_ctype(dtype), False, False)
    body = relu_body(dtype)
    return f"""{header}
{relu_fn}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def make_is_finite(dtype: DTypeSpec) -> str:
    name = f"is_finite_scalar_{dtype.name}"
    header = header_push()
    buffers = (
        "struct IsFiniteScalarBuffers<T> {\n"
        "    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;\n"
        "    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;\n"
        "    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;\n"
        "};\n\n"
        f"ParameterBlock<IsFiniteScalarBuffers<{dtype_to_ctype(dtype)}>> gIsFinite_{dtype_to_ctype(dtype)};\n"
    )
    body = is_finite_body(dtype)
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(1, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def make_accumulate(op: str, in_name: str, out_name: str) -> str:
    name = f"{op}_{in_name}_{out_name}"
    in_kind, in_bits = packed_kind(in_name)
    if in_kind:
        header = header_push()
        buffers = (
            f"struct {op.capitalize()}AccBuffers {{\n"
            "    [[vk::binding(0, 0)]] ByteAddressBuffer input0;\n"
            "    [[vk::binding(1, 0)]] ByteAddressBuffer input1;\n"
            f"    [[vk::binding(2, 0)]] RWStructuredBuffer<{acc_ctype(out_name)}, Std430DataLayout> output0;\n"
            "};\n\n"
            f"ParameterBlock<{op.capitalize()}AccBuffers> g{op.capitalize()}Acc;\n"
        )
        read_fn = "packed_read_signed" if in_kind == "signed" else "packed_read_unsigned"
        cast = "int" if in_kind == "signed" else "uint"
        op_expr = "+" if op == "add" else "*"
        body = f"""uint i = tid.x;
if (i < pc.len) {{
    {cast} a = {read_fn}(g{op.capitalize()}Acc.input0, i, {in_bits});
    {cast} b = {read_fn}(g{op.capitalize()}Acc.input1, i, {in_bits});
    g{op.capitalize()}Acc.output0[i] = ({acc_ctype(out_name)})({cast})(a {op_expr} b);
}}
"""
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    header = header_push()
    buffers = (
        f"struct {op.capitalize()}AccBuffers {{\n"
        f"    [[vk::binding(0, 0)]] StructuredBuffer<{acc_ctype(in_name)}, Std430DataLayout> input0;\n"
        f"    [[vk::binding(1, 0)]] StructuredBuffer<{acc_ctype(in_name)}, Std430DataLayout> input1;\n"
        f"    [[vk::binding(2, 0)]] RWStructuredBuffer<{acc_ctype(out_name)}, Std430DataLayout> output0;\n"
        "};\n\n"
        f"ParameterBlock<{op.capitalize()}AccBuffers> g{op.capitalize()}Acc;\n"
    )
    op_expr = "+" if op == "add" else "*"
    body = f"""uint i = tid.x;
if (i < pc.len) {{
    {acc_ctype(out_name)} a = ({acc_ctype(out_name)})g{op.capitalize()}Acc.input0[i];
    {acc_ctype(out_name)} b = ({acc_ctype(out_name)})g{op.capitalize()}Acc.input1[i];
    g{op.capitalize()}Acc.output0[i] = a {op_expr} b;
}}
"""
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def make_abs_accumulate(in_name: str, out_name: str) -> str:
    name = f"abs_{in_name}_{out_name}"
    in_kind, in_bits = packed_kind(in_name)
    header = header_push()
    if in_kind:
        buffers = (
            "struct AbsAccBuffers {\n"
            "    [[vk::binding(0, 0)]] ByteAddressBuffer input0;\n"
            f"    [[vk::binding(2, 0)]] RWStructuredBuffer<{acc_ctype(out_name)}, Std430DataLayout> output0;\n"
            "};\n\n"
            "ParameterBlock<AbsAccBuffers> gAbsAcc;\n"
        )
        body = f"""uint i = tid.x;
if (i < pc.len) {{
    int x = packed_read_signed(gAbsAcc.input0, i, {in_bits});
    int y = x < 0 ? -x : x;
    gAbsAcc.output0[i] = ({acc_ctype(out_name)})y;
}}
"""
        return f"""{include_packed()}{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""
    buffers = (
        "struct AbsAccBuffers {\n"
        f"    [[vk::binding(0, 0)]] StructuredBuffer<{acc_ctype(in_name)}, Std430DataLayout> input0;\n"
        f"    [[vk::binding(2, 0)]] RWStructuredBuffer<{acc_ctype(out_name)}, Std430DataLayout> output0;\n"
        "};\n\n"
        "ParameterBlock<AbsAccBuffers> gAbsAcc;\n"
    )
    body = f"""uint i = tid.x;
if (i < pc.len) {{
    {acc_ctype(out_name)} x = ({acc_ctype(out_name)})gAbsAcc.input0[i];
    gAbsAcc.output0[i] = x < 0 ? -x : x;
}}
"""
    return f"""{header}
{buffers}
[shader("compute")]
[numthreads(256, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {{
{body}}}
"""


def acc_ctype(name: str) -> str:
    if name.startswith("i"):
        return "int64_t" if name == "i64" else "int"
    if name.startswith("u"):
        return "uint64_t" if name == "u64" else "uint"
    return "int"


def packed_kind(name: str) -> tuple[str | None, int]:
    if name.startswith("i") and len(name) == 2 and name[1] in {"1", "2", "4"}:
        return "signed", int(name[1])
    if name.startswith("u") and len(name) == 2 and name[1] in {"1", "2", "4"}:
        return "unsigned", int(name[1])
    return (None, 0)


def write_op(op: str, base: Iterable[DTypeSpec], inplace: bool, accumulate: bool) -> None:
    base_dir = VULKAN_DIR / op / "shaders" / "base"
    inplace_dir = VULKAN_DIR / op / "shaders" / "inplace"
    acc_dir = VULKAN_DIR / op / "shaders" / "accumulate"
    ensure_dir(base_dir)
    if inplace:
        ensure_dir(inplace_dir)
    if accumulate:
        ensure_dir(acc_dir)

    for dtype in base:
        if op in ("add", "mul"):
            write_file(base_dir / f"{op}_{dtype.name}.slang", make_add_mul(op, False, dtype))
            if inplace:
                write_file(
                    inplace_dir / f"{op}_inplace_{dtype.name}.slang",
                    make_add_mul(op, True, dtype),
                )
        elif op == "abs":
            write_file(base_dir / f"abs_{dtype.name}.slang", make_abs(False, dtype))
            if inplace:
                write_file(
                    inplace_dir / f"abs_inplace_{dtype.name}.slang",
                    make_abs(True, dtype),
                )
        elif op == "fill":
            write_file(base_dir / f"fill_{dtype.name}.slang", make_fill(False, dtype))
            if inplace:
                write_file(
                    inplace_dir / f"fill_inplace_{dtype.name}.slang",
                    make_fill(True, dtype),
                )
        elif op == "relu":
            write_file(base_dir / f"relu_{dtype.name}.slang", make_relu(dtype))
        elif op == "is_finite":
            write_file(base_dir / f"is_finite_scalar_{dtype.name}.slang", make_is_finite(dtype))

    if accumulate:
        if op in ("add", "mul"):
            for in_name, out_name in ACC_SIGNED + ACC_UNSIGNED + ACC_PACKED_SIGNED + ACC_PACKED_UNSIGNED:
                write_file(
                    acc_dir / f"{op}_{in_name}_{out_name}.slang",
                    make_accumulate(op, in_name, out_name),
                )
        if op == "abs":
            for in_name, out_name in ACC_SIGNED + ACC_PACKED_SIGNED:
                write_file(
                    acc_dir / f"abs_{in_name}_{out_name}.slang",
                    make_abs_accumulate(in_name, out_name),
                )


def main() -> None:
    write_op("add", BASE_ADD_MUL_FILL, inplace=True, accumulate=True)
    write_op("mul", BASE_ADD_MUL_FILL, inplace=True, accumulate=True)
    write_op("abs", BASE_ABS, inplace=True, accumulate=True)
    write_op("fill", BASE_ADD_MUL_FILL, inplace=True, accumulate=False)
    write_op("relu", BASE_RELU, inplace=False, accumulate=False)
    write_op("is_finite", BASE_IS_FINITE, inplace=False, accumulate=False)
    write_matmul()


def write_matmul() -> None:
    base = (
        FLOAT_DTYPES
        + SIGNED_DTYPES
        + UNSIGNED_DTYPES
        + BOOL_DTYPE
        + BITSET_DTYPE
        + PACKED_SIGNED
        + PACKED_UNSIGNED
    )
    base_dir = VULKAN_DIR / "matmul" / "shaders" / "base"
    inplace_dir = VULKAN_DIR / "matmul" / "shaders" / "inplace"
    acc_dir = VULKAN_DIR / "matmul" / "shaders" / "accumulate"
    ensure_dir(base_dir)
    ensure_dir(inplace_dir)
    ensure_dir(acc_dir)

    for dtype in base:
        write_file(base_dir / f"matmul_{dtype.name}.slang", matmul_base("matmul", dtype, False))
        write_file(
            inplace_dir / f"matmul_inplace_{dtype.name}.slang",
            matmul_base("matmul", dtype, True),
        )

    for in_name, out_name in ACC_SIGNED + ACC_UNSIGNED + ACC_PACKED_SIGNED + ACC_PACKED_UNSIGNED:
        signed = in_name.startswith("i")
        write_file(
            acc_dir / f"matmul_{in_name}_{out_name}.slang",
            matmul_accumulate(in_name, out_name, signed),
        )


if __name__ == "__main__":
    main()
