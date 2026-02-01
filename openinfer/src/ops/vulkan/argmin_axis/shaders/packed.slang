#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "../../shaders/reduce_utils.slang"
#include "common.slang"

#ifndef HAS_I64
#define HAS_I64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { ArgminAxisPush push; }

#if HAS_I64
#define ARGMIN_PACKED_SIGNED(NAME) \
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
        int64_t best_idx = 0; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                int v = load_packed_i(input_desc, i, data); \
                int64_t axis_idx = int64_t(in_coords[push.axis]); \
                bool better = push.select_first != 0u ? (v < best) : (v <= best); \
                if (!initialized || better) { \
                    best = v; \
                    best_idx = axis_idx; \
                    initialized = true; \
                } \
            } \
        } \
        store_i64(out_desc, out_idx, best_idx, out_buf); \
    }

#define ARGMIN_PACKED_UNSIGNED(NAME) \
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
        int64_t best_idx = 0; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                uint v = load_packed_u(input_desc, i, data); \
                int64_t axis_idx = int64_t(in_coords[push.axis]); \
                bool better = push.select_first != 0u ? (v < best) : (v <= best); \
                if (!initialized || better) { \
                    best = v; \
                    best_idx = axis_idx; \
                    initialized = true; \
                } \
            } \
        } \
        store_i64(out_desc, out_idx, best_idx, out_buf); \
    }

ARGMIN_PACKED_SIGNED(argmin_axis_i1_packed)
ARGMIN_PACKED_SIGNED(argmin_axis_i2_packed)
ARGMIN_PACKED_SIGNED(argmin_axis_i4_packed)
ARGMIN_PACKED_UNSIGNED(argmin_axis_u1_packed)
ARGMIN_PACKED_UNSIGNED(argmin_axis_u2_packed)
ARGMIN_PACKED_UNSIGNED(argmin_axis_u4_packed)

#undef ARGMIN_PACKED_SIGNED
#undef ARGMIN_PACKED_UNSIGNED
#endif
