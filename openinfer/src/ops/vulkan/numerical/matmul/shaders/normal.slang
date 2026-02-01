#include "../../../shaders/common.slang"
#include "../../../shaders/float_utils.slang"
#include "common.slang"

#ifndef HAS_F16
#define HAS_F16 0
#endif

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

#ifndef HAS_F64
#define HAS_F64 0
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

float load_f32_elem(TensorDesc desc, uint elem_offset) {
    return asfloat(data.Load(elem_byte_offset(desc, elem_offset)));
}

void store_f32_elem(TensorDesc desc, uint elem_offset, float v) {
    out_buf.Store(elem_byte_offset(desc, elem_offset), asuint(v));
}

float load_f16_elem(TensorDesc desc, uint elem_offset) {
    return f16_to_f32(load_u16(data, elem_byte_offset(desc, elem_offset)));
}

void store_f16_elem(TensorDesc desc, uint elem_offset, float v) {
    store_u16(out_buf, elem_byte_offset(desc, elem_offset), f32_to_f16(v));
}

float load_bf16_elem(TensorDesc desc, uint elem_offset) {
    return bf16_to_f32(load_u16(data, elem_byte_offset(desc, elem_offset)));
}

void store_bf16_elem(TensorDesc desc, uint elem_offset, float v) {
    store_u16(out_buf, elem_byte_offset(desc, elem_offset), f32_to_bf16(v));
}

float load_f8_elem(TensorDesc desc, uint elem_offset) {
    return f8_to_f32(load_u8(data, elem_byte_offset(desc, elem_offset)));
}

void store_f8_elem(TensorDesc desc, uint elem_offset, float v) {
    store_u8(out_buf, elem_byte_offset(desc, elem_offset), f32_to_f8(v));
}

int load_i32_elem(TensorDesc desc, uint elem_offset) {
    return asint(data.Load(elem_byte_offset(desc, elem_offset)));
}

void store_i32_elem(TensorDesc desc, uint elem_offset, int v) {
    out_buf.Store(elem_byte_offset(desc, elem_offset), asuint(v));
}

uint load_u32_elem(TensorDesc desc, uint elem_offset) {
    return data.Load(elem_byte_offset(desc, elem_offset));
}

void store_u32_elem(TensorDesc desc, uint elem_offset, uint v) {
    out_buf.Store(elem_byte_offset(desc, elem_offset), v);
}

int load_i16_elem(TensorDesc desc, uint elem_offset) {
    return int(load_u16(data, elem_byte_offset(desc, elem_offset)));
}

void store_i16_elem(TensorDesc desc, uint elem_offset, int v) {
    store_u16(out_buf, elem_byte_offset(desc, elem_offset), uint(v));
}

uint load_u16_elem(TensorDesc desc, uint elem_offset) {
    return load_u16(data, elem_byte_offset(desc, elem_offset));
}

void store_u16_elem(TensorDesc desc, uint elem_offset, uint v) {
    store_u16(out_buf, elem_byte_offset(desc, elem_offset), v);
}

int load_i8_elem(TensorDesc desc, uint elem_offset) {
    return int(load_u8(data, elem_byte_offset(desc, elem_offset)));
}

void store_i8_elem(TensorDesc desc, uint elem_offset, int v) {
    store_u8(out_buf, elem_byte_offset(desc, elem_offset), uint(v));
}

uint load_u8_elem(TensorDesc desc, uint elem_offset) {
    return load_u8(data, elem_byte_offset(desc, elem_offset));
}

void store_u8_elem(TensorDesc desc, uint elem_offset, uint v) {
    store_u8(out_buf, elem_byte_offset(desc, elem_offset), v);
}

uint load_bool_elem(TensorDesc desc, uint elem_offset) {
    return load_bool(data, elem_byte_offset(desc, elem_offset));
}

void store_bool_elem(TensorDesc desc, uint elem_offset, uint v) {
    store_bool(out_buf, elem_byte_offset(desc, elem_offset), v);
}

uint load_bitset_elem(TensorDesc desc, uint elem_offset) {
    return load_bitset(data, elem_byte_offset(desc, elem_offset));
}

void store_bitset_elem(TensorDesc desc, uint elem_offset, uint v) {
    store_bitset(out_buf, elem_byte_offset(desc, elem_offset), v);
}

#if HAS_I64
int64_t load_i64_elem(TensorDesc desc, uint elem_offset) {
    uint offset = elem_byte_offset(desc, elem_offset);
    uint lo = data.Load(offset);
    uint hi = data.Load(offset + 4u);
    uint64_t value = (uint64_t(hi) << 32u) | uint64_t(lo);
    return int64_t(value);
}

void store_i64_elem(TensorDesc desc, uint elem_offset, int64_t v) {
    uint offset = elem_byte_offset(desc, elem_offset);
    uint64_t value = uint64_t(v);
    out_buf.Store(offset, uint(value & 0xFFFFFFFFu));
    out_buf.Store(offset + 4u, uint((value >> 32u) & 0xFFFFFFFFu));
}
#endif

#if HAS_U64
uint64_t load_u64_elem(TensorDesc desc, uint elem_offset) {
    uint offset = elem_byte_offset(desc, elem_offset);
    uint lo = data.Load(offset);
    uint hi = data.Load(offset + 4u);
    return (uint64_t(hi) << 32u) | uint64_t(lo);
}

void store_u64_elem(TensorDesc desc, uint elem_offset, uint64_t v) {
    uint offset = elem_byte_offset(desc, elem_offset);
    out_buf.Store(offset, uint(v & 0xFFFFFFFFu));
    out_buf.Store(offset + 4u, uint((v >> 32u) & 0xFFFFFFFFu));
}
#endif

#if HAS_F64
double load_f64_elem(TensorDesc desc, uint elem_offset) {
    uint offset = elem_byte_offset(desc, elem_offset);
    uint lo = data.Load(offset);
    uint hi = data.Load(offset + 4u);
    return asdouble(uint2(lo, hi));
}

void store_f64_elem(TensorDesc desc, uint elem_offset, double v) {
    uint offset = elem_byte_offset(desc, elem_offset);
    uint2 bits = asuint(v);
    out_buf.Store(offset, bits.x);
    out_buf.Store(offset + 4u, bits.y);
}
#endif

void matmul_indices(uint idx, out uint batch_rank, out uint batch_linear, out uint m, out uint n) {
    TensorDesc out = tensor_descs[2];
    uint out_rank = out.rank;
    batch_rank = out_rank - 2u;
    uint out_n = out.shape[out_rank - 1u];
    uint out_m = out.shape[out_rank - 2u];
    n = idx % out_n;
    uint tmp = idx / out_n;
    m = tmp % out_m;
    batch_linear = tmp / out_m;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    float acc = 0.0f;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_f32_elem(a, a_offset) * load_f32_elem(b, b_offset);
    }
    store_f32_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_f32_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    float acc = 0.0f;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_f16_elem(a, a_offset) * load_f16_elem(b, b_offset);
    }
    store_f16_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_f16_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    float acc = 0.0f;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_bf16_elem(a, a_offset) * load_bf16_elem(b, b_offset);
    }
    store_bf16_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_bf16_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    float acc = 0.0f;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_f8_elem(a, a_offset) * load_f8_elem(b, b_offset);
    }
    store_f8_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_f8_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    int acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_i32_elem(a, a_offset) * load_i32_elem(b, b_offset);
    }
    store_i32_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i32_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    int acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_i16_elem(a, a_offset) * load_i16_elem(b, b_offset);
    }
    store_i16_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i16_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    int acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_i8_elem(a, a_offset) * load_i8_elem(b, b_offset);
    }
    store_i8_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i8_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    uint acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_u32_elem(a, a_offset) * load_u32_elem(b, b_offset);
    }
    store_u32_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u32_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    uint acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_u16_elem(a, a_offset) * load_u16_elem(b, b_offset);
    }
    store_u16_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u16_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    uint acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_u8_elem(a, a_offset) * load_u8_elem(b, b_offset);
    }
    store_u8_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u8_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_bool_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    uint acc = 0u;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        uint prod = load_bool_elem(a, a_offset) & load_bool_elem(b, b_offset);
        acc = (acc | prod);
    }
    store_bool_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_bool_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_bool_normal(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_bitset_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    uint acc = 0u;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        uint prod = load_bitset_elem(a, a_offset) * load_bitset_elem(b, b_offset);
        acc += prod;
    }
    store_bitset_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_bitset_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_bitset_normal(tid);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    int64_t acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_i64_elem(a, a_offset) * load_i64_elem(b, b_offset);
    }
    store_i64_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i64_normal(tid);
}
#endif

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    uint64_t acc = 0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_u64_elem(a, a_offset) * load_u64_elem(b, b_offset);
    }
    store_u64_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u64_normal(tid);
}
#endif

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint batch_rank, batch_linear, m, n;
    matmul_indices(idx, batch_rank, batch_linear, m, n);
    uint a_base = batch_base_offset(a, batch_rank, batch_linear);
    uint b_base = batch_base_offset(b, batch_rank, batch_linear);
    uint out_base = batch_base_offset(out, batch_rank, batch_linear);
    uint a_stride_m = a.strides[batch_rank];
    uint a_stride_k = a.strides[batch_rank + 1u];
    uint b_stride_k = b.strides[batch_rank];
    uint b_stride_n = b.strides[batch_rank + 1u];
    uint out_stride_m = out.strides[batch_rank];
    uint out_stride_n = out.strides[batch_rank + 1u];
    uint a_m_offset = a_base + m * a_stride_m;
    uint out_offset = out_base + m * out_stride_m + n * out_stride_n;
    double acc = 0.0;
    uint k_len = a.shape[batch_rank + 1u];
    for (uint k = 0; k < k_len; k++) {
        uint a_offset = a_m_offset + k * a_stride_k;
        uint b_offset = b_base + k * b_stride_k + n * b_stride_n;
        acc += load_f64_elem(a, a_offset) * load_f64_elem(b, b_offset);
    }
    store_f64_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_f64_normal(tid);
}
#endif
