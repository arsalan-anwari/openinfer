#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "../../shaders/reduce_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { SumAxisPush push; }

#define SUM_AXIS_PACKED_SIGNED(NAME) \
    [numthreads(256, 1, 1)] \
    [shader("compute")] \
    void NAME(uint3 tid : SV_DispatchThreadID) { \
        uint out_idx = tid.x; \
        if (out_idx >= push.output_len) return; \
        TensorDesc input_desc = tensor_descs[0]; \
        TensorDesc out_desc = tensor_descs[1]; \
        uint out_coords[OPENINFER_VK_MAX_DIMS]; \
        uint in_coords[OPENINFER_VK_MAX_DIMS]; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        int acc = 0; \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                acc += load_packed_i(input_desc, i, data); \
            } \
        } \
        store_packed(out_desc, out_idx, uint(acc), out_buf); \
    }

#define SUM_AXIS_PACKED_UNSIGNED(NAME) \
    [numthreads(256, 1, 1)] \
    [shader("compute")] \
    void NAME(uint3 tid : SV_DispatchThreadID) { \
        uint out_idx = tid.x; \
        if (out_idx >= push.output_len) return; \
        TensorDesc input_desc = tensor_descs[0]; \
        TensorDesc out_desc = tensor_descs[1]; \
        uint out_coords[OPENINFER_VK_MAX_DIMS]; \
        uint in_coords[OPENINFER_VK_MAX_DIMS]; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        uint acc = 0u; \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                acc += load_packed_u(input_desc, i, data); \
            } \
        } \
        store_packed(out_desc, out_idx, acc, out_buf); \
    }

SUM_AXIS_PACKED_SIGNED(sum_axis_i1_packed)
SUM_AXIS_PACKED_SIGNED(sum_axis_i2_packed)
SUM_AXIS_PACKED_SIGNED(sum_axis_i4_packed)
SUM_AXIS_PACKED_UNSIGNED(sum_axis_u1_packed)
SUM_AXIS_PACKED_UNSIGNED(sum_axis_u2_packed)
SUM_AXIS_PACKED_UNSIGNED(sum_axis_u4_packed)

#undef SUM_AXIS_PACKED_SIGNED
#undef SUM_AXIS_PACKED_UNSIGNED
