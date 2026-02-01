#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "../../shaders/reduce_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { MinAxisPush push; }

#define MIN_AXIS_PACKED_SIGNED(NAME) \
    [numthreads(256, 1, 1)] \
    [shader("compute")] \
    void NAME(uint3 tid : SV_DispatchThreadID) { \
        uint out_idx = tid.x; \
        if (out_idx >= push.output_len) return; \
        TensorDesc input_desc = tensor_descs[0]; \
        TensorDesc out_desc = tensor_descs[1]; \
        uint out_coords[OPENINFER_VK_MAX_DIMS]; \
        uint in_coords[OPENINFER_VK_MAX_DIMS]; \
        bool initialized = false; \
        int best = 0; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                int v = load_packed_i(input_desc, i, data); \
                if (!initialized || v < best) { \
                    best = v; \
                    initialized = true; \
                } \
            } \
        } \
        store_packed(out_desc, out_idx, uint(best), out_buf); \
    }

#define MIN_AXIS_PACKED_UNSIGNED(NAME) \
    [numthreads(256, 1, 1)] \
    [shader("compute")] \
    void NAME(uint3 tid : SV_DispatchThreadID) { \
        uint out_idx = tid.x; \
        if (out_idx >= push.output_len) return; \
        TensorDesc input_desc = tensor_descs[0]; \
        TensorDesc out_desc = tensor_descs[1]; \
        uint out_coords[OPENINFER_VK_MAX_DIMS]; \
        uint in_coords[OPENINFER_VK_MAX_DIMS]; \
        bool initialized = false; \
        uint best = 0u; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                uint v = load_packed_u(input_desc, i, data); \
                if (!initialized || v < best) { \
                    best = v; \
                    initialized = true; \
                } \
            } \
        } \
        store_packed(out_desc, out_idx, best, out_buf); \
    }

MIN_AXIS_PACKED_SIGNED(min_axis_i1_packed)
MIN_AXIS_PACKED_SIGNED(min_axis_i2_packed)
MIN_AXIS_PACKED_SIGNED(min_axis_i4_packed)
MIN_AXIS_PACKED_UNSIGNED(min_axis_u1_packed)
MIN_AXIS_PACKED_UNSIGNED(min_axis_u2_packed)
MIN_AXIS_PACKED_UNSIGNED(min_axis_u4_packed)

#undef MIN_AXIS_PACKED_SIGNED
#undef MIN_AXIS_PACKED_UNSIGNED
