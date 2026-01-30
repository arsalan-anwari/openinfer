#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { MatmulPush push; }

uint packed_bit_offset_elem(TensorDesc desc, uint elem_offset) {
    return desc.byte_offset * 8u + elem_offset * desc.elem_bits;
}

int load_packed_i_elem(TensorDesc desc, uint elem_offset) {
    uint bits = read_bits(data, packed_bit_offset_elem(desc, elem_offset), desc.elem_bits);
    return sign_extend(bits, desc.elem_bits);
}

uint load_packed_u_elem(TensorDesc desc, uint elem_offset) {
    return read_bits(data, packed_bit_offset_elem(desc, elem_offset), desc.elem_bits);
}

void store_packed_elem(TensorDesc desc, uint elem_offset, uint v) {
    write_bits(out_buf, packed_bit_offset_elem(desc, elem_offset), desc.elem_bits, v);
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

void matmul_packed_signed(uint idx) {
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
        int lhs = load_packed_i_elem(a, a_offset);
        int rhs = load_packed_i_elem(b, b_offset);
        acc += lhs * rhs;
        acc = (acc << 24) >> 24;
    }
    store_packed_elem(out, out_offset, uint(acc));
}

void matmul_packed_unsigned(uint idx) {
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
        uint lhs = load_packed_u_elem(a, a_offset);
        uint rhs = load_packed_u_elem(b, b_offset);
        acc = (acc + (lhs * rhs)) & 0xFFu;
    }
    store_packed_elem(out, out_offset, acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    matmul_packed_signed(idx);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i4_packed(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    matmul_packed_signed(idx);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i2_packed(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    matmul_packed_signed(idx);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_i1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_i1_packed(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    matmul_packed_unsigned(idx);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u4_packed(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    matmul_packed_unsigned(idx);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u2_packed(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    matmul_packed_unsigned(idx);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void matmul_u1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    matmul_u1_packed(tid);
}
