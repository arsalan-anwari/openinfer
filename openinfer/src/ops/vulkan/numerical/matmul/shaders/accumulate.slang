#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
#include "common.slang"

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { MatmulPush push; }

uint elem_byte_offset(TensorDesc desc, uint elem_offset) {
    return desc.byte_offset + elem_offset * elem_byte_stride(desc);
}

uint batch_base_offset(TensorDesc desc, uint batch_rank, uint batch_linear) {
    uint offset = 0;
    uint linear = batch_linear;
    for (int dim = int(batch_rank) - 1; dim >= 0; --dim) {
        uint dim_size = desc.shape[dim];
        uint coord = (dim_size == 0u) ? 0u : (linear % dim_size);
        linear = (dim_size == 0u) ? 0u : (linear / dim_size);
        uint dim_extent = desc.shape[dim];
        uint use_coord = (dim_extent == 1u) ? 0u : coord;
        offset += use_coord * desc.strides[dim];
    }
    return offset;
}

void matmul_setup(uint idx, out uint n, out uint out_offset, out uint a_m_offset, out uint b_base, out uint a_stride_k, out uint b_stride_k, out uint b_stride_n, out uint k_len) {
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint out_rank = out.rank;
    uint batch_rank = out_rank - 2u;
    uint out_n = out.shape[out_rank - 1u];
    uint out_m = out.shape[out_rank - 2u];
    n = idx % out_n;
    uint tmp = idx / out_n;
    uint m = tmp % out_m;
    uint batch_linear = tmp / out_m;
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    a_stride_k = a.strides[batch_rank + 1u];
    b_stride_k = b.strides[batch_rank];
    b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    a_m_offset = a_base + m * a_stride_m;
    out_offset = out_base + m * out_stride_m + n * out_stride_n;
    k_len = a.shape[batch_rank + 1u];
}

int load_i8_elem(TensorDesc desc, uint elem_offset) {
    return int(load_u8(data, elem_byte_offset(desc, elem_offset)));
}

int load_i16_elem(TensorDesc desc, uint elem_offset) {
    return int(load_u16(data, elem_byte_offset(desc, elem_offset)));
}

int load_i32_elem(TensorDesc desc, uint elem_offset) {
    return asint(data.Load(elem_byte_offset(desc, elem_offset)));
}

uint load_u8_elem(TensorDesc desc, uint elem_offset) {
    return load_u8(data, elem_byte_offset(desc, elem_offset));
}

uint load_u16_elem(TensorDesc desc, uint elem_offset) {
    return load_u16(data, elem_byte_offset(desc, elem_offset));
}

uint load_u32_elem(TensorDesc desc, uint elem_offset) {
    return data.Load(elem_byte_offset(desc, elem_offset));
}

int load_packed_i_elem(TensorDesc desc, uint elem_offset) {
    uint bits = read_bits(data, desc.byte_offset * 8u + elem_offset * desc.elem_bits, desc.elem_bits);
    return sign_extend(bits, desc.elem_bits);
}

uint load_packed_u_elem(TensorDesc desc, uint elem_offset) {
    return read_bits(data, desc.byte_offset * 8u + elem_offset * desc.elem_bits, desc.elem_bits);
}

void store_i8_elem(TensorDesc desc, uint elem_offset, int v) {
    store_u8(out_buf, elem_byte_offset(desc, elem_offset), uint(v));
}

void store_i16_elem(TensorDesc desc, uint elem_offset, int v) {
    store_u16(out_buf, elem_byte_offset(desc, elem_offset), uint(v));
}

void store_i32_elem(TensorDesc desc, uint elem_offset, int v) {
    out_buf.Store(elem_byte_offset(desc, elem_offset), asuint(v));
}

void store_u8_elem(TensorDesc desc, uint elem_offset, uint v) {
    store_u8(out_buf, elem_byte_offset(desc, elem_offset), v);
}

void store_u16_elem(TensorDesc desc, uint elem_offset, uint v) {
    store_u16(out_buf, elem_byte_offset(desc, elem_offset), v);
}

void store_u32_elem(TensorDesc desc, uint elem_offset, uint v) {
    out_buf.Store(elem_byte_offset(desc, elem_offset), v);
}

#if HAS_I64
void store_i64_elem(TensorDesc desc, uint elem_offset, int64_t v) {
    uint offset = elem_byte_offset(desc, elem_offset);
    uint64_t value = uint64_t(v);
    out_buf.Store(offset, uint(value & 0xFFFFFFFFu));
    out_buf.Store(offset + 4u, uint((value >> 32u) & 0xFFFFFFFFu));
}
#endif

#if HAS_U64
void store_u64_elem(TensorDesc desc, uint elem_offset, uint64_t v) {
    uint offset = elem_byte_offset(desc, elem_offset);
    out_buf.Store(offset, uint(v & 0xFFFFFFFFu));
    out_buf.Store(offset + 4u, uint((v >> 32u) & 0xFFFFFFFFu));
}
#endif

// Signed accumulate (packed + non-packed)
#define MATMUL_SIGNED_ACCUM(fn_name, load_a, load_b, store_fn) \
[numthreads(256, 1, 1)] \
[shader(\"compute\")] \
void fn_name(uint3 tid : SV_DispatchThreadID) { \
    uint idx = tid.x; \
    if (idx >= push.len) return; \
    uint n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len; \
    matmul_setup(idx, n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len); \
    int64_t acc = 0; \
    for (uint k = 0; k < k_len; k++) { \
        uint a_offset = a_m_offset + k * a_stride_k; \
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n; \
        int lhs = load_a; \
        int rhs = load_b; \
        acc += int64_t(lhs) * int64_t(rhs); \
    } \
    store_fn(tensor_descs[2], out_offset, int(acc)); \
}

#define MATMUL_SIGNED_ACCUM_I64(fn_name, load_a, load_b) \
[numthreads(256, 1, 1)] \
[shader(\"compute\")] \
void fn_name(uint3 tid : SV_DispatchThreadID) { \
    uint idx = tid.x; \
    if (idx >= push.len) return; \
    uint n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len; \
    matmul_setup(idx, n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len); \
    int64_t acc = 0; \
    for (uint k = 0; k < k_len; k++) { \
        uint a_offset = a_m_offset + k * a_stride_k; \
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n; \
        int lhs = load_a; \
        int rhs = load_b; \
        acc += int64_t(lhs) * int64_t(rhs); \
    } \
    store_i64_elem(tensor_descs[2], out_offset, acc); \
}

MATMUL_SIGNED_ACCUM(matmul_i1_accumulate_i8, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i8_elem)
MATMUL_SIGNED_ACCUM(matmul_i2_accumulate_i8, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i8_elem)
MATMUL_SIGNED_ACCUM(matmul_i4_accumulate_i8, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i8_elem)
MATMUL_SIGNED_ACCUM(matmul_i1_accumulate_i16, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i16_elem)
MATMUL_SIGNED_ACCUM(matmul_i2_accumulate_i16, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i16_elem)
MATMUL_SIGNED_ACCUM(matmul_i4_accumulate_i16, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i16_elem)
MATMUL_SIGNED_ACCUM(matmul_i1_accumulate_i32, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i32_elem)
MATMUL_SIGNED_ACCUM(matmul_i2_accumulate_i32, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i32_elem)
MATMUL_SIGNED_ACCUM(matmul_i4_accumulate_i32, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset), store_i32_elem)
MATMUL_SIGNED_ACCUM(matmul_i8_accumulate_i16, load_i8_elem(tensor_descs[0], a_offset), load_i8_elem(tensor_descs[1], b_offset), store_i16_elem)
MATMUL_SIGNED_ACCUM(matmul_i8_accumulate_i32, load_i8_elem(tensor_descs[0], a_offset), load_i8_elem(tensor_descs[1], b_offset), store_i32_elem)
MATMUL_SIGNED_ACCUM(matmul_i16_accumulate_i32, load_i16_elem(tensor_descs[0], a_offset), load_i16_elem(tensor_descs[1], b_offset), store_i32_elem)

#if HAS_I64
MATMUL_SIGNED_ACCUM_I64(matmul_i1_accumulate_i64, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset))
MATMUL_SIGNED_ACCUM_I64(matmul_i2_accumulate_i64, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset))
MATMUL_SIGNED_ACCUM_I64(matmul_i4_accumulate_i64, load_packed_i_elem(tensor_descs[0], a_offset), load_packed_i_elem(tensor_descs[1], b_offset))
MATMUL_SIGNED_ACCUM_I64(matmul_i8_accumulate_i64, load_i8_elem(tensor_descs[0], a_offset), load_i8_elem(tensor_descs[1], b_offset))
MATMUL_SIGNED_ACCUM_I64(matmul_i16_accumulate_i64, load_i16_elem(tensor_descs[0], a_offset), load_i16_elem(tensor_descs[1], b_offset))
MATMUL_SIGNED_ACCUM_I64(matmul_i32_accumulate_i64, load_i32_elem(tensor_descs[0], a_offset), load_i32_elem(tensor_descs[1], b_offset))
#endif

// Unsigned accumulate
#define MATMUL_UNSIGNED_ACCUM(fn_name, load_a, load_b, store_fn) \
[numthreads(256, 1, 1)] \
[shader(\"compute\")] \
void fn_name(uint3 tid : SV_DispatchThreadID) { \
    uint idx = tid.x; \
    if (idx >= push.len) return; \
    uint n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len; \
    matmul_setup(idx, n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len); \
    uint64_t acc = 0; \
    for (uint k = 0; k < k_len; k++) { \
        uint a_offset = a_m_offset + k * a_stride_k; \
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n; \
        uint lhs = load_a; \
        uint rhs = load_b; \
        acc += uint64_t(lhs) * uint64_t(rhs); \
    } \
    store_fn(tensor_descs[2], out_offset, uint(acc)); \
}

#define MATMUL_UNSIGNED_ACCUM_U64(fn_name, load_a, load_b) \
[numthreads(256, 1, 1)] \
[shader(\"compute\")] \
void fn_name(uint3 tid : SV_DispatchThreadID) { \
    uint idx = tid.x; \
    if (idx >= push.len) return; \
    uint n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len; \
    matmul_setup(idx, n, out_offset, a_m_offset, b_base, a_stride_k, b_stride_k, b_stride_n, k_len); \
    uint64_t acc = 0; \
    for (uint k = 0; k < k_len; k++) { \
        uint a_offset = a_m_offset + k * a_stride_k; \
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n; \
        uint lhs = load_a; \
        uint rhs = load_b; \
        acc += uint64_t(lhs) * uint64_t(rhs); \
    } \
    store_u64_elem(tensor_descs[2], out_offset, acc); \
}

MATMUL_UNSIGNED_ACCUM(matmul_u1_accumulate_u8, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u8_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u2_accumulate_u8, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u8_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u4_accumulate_u8, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u8_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u1_accumulate_u16, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u16_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u2_accumulate_u16, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u16_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u4_accumulate_u16, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u16_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u1_accumulate_u32, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u32_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u2_accumulate_u32, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u32_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u4_accumulate_u32, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset), store_u32_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u8_accumulate_u16, load_u8_elem(tensor_descs[0], a_offset), load_u8_elem(tensor_descs[1], b_offset), store_u16_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u8_accumulate_u32, load_u8_elem(tensor_descs[0], a_offset), load_u8_elem(tensor_descs[1], b_offset), store_u32_elem)
MATMUL_UNSIGNED_ACCUM(matmul_u16_accumulate_u32, load_u16_elem(tensor_descs[0], a_offset), load_u16_elem(tensor_descs[1], b_offset), store_u32_elem)

#if HAS_U64
MATMUL_UNSIGNED_ACCUM_U64(matmul_u1_accumulate_u64, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset))
MATMUL_UNSIGNED_ACCUM_U64(matmul_u2_accumulate_u64, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset))
MATMUL_UNSIGNED_ACCUM_U64(matmul_u4_accumulate_u64, load_packed_u_elem(tensor_descs[0], a_offset), load_packed_u_elem(tensor_descs[1], b_offset))
MATMUL_UNSIGNED_ACCUM_U64(matmul_u8_accumulate_u64, load_u8_elem(tensor_descs[0], a_offset), load_u8_elem(tensor_descs[1], b_offset))
MATMUL_UNSIGNED_ACCUM_U64(matmul_u16_accumulate_u64, load_u16_elem(tensor_descs[0], a_offset), load_u16_elem(tensor_descs[1], b_offset))
MATMUL_UNSIGNED_ACCUM_U64(matmul_u32_accumulate_u64, load_u32_elem(tensor_descs[0], a_offset), load_u32_elem(tensor_descs[1], b_offset))
#endif
