#include "../../shaders/common.slang"
#include "../../shaders/reduce_utils.slang"
#include "common.slang"

#ifndef HAS_F64
#define HAS_F64 0
#endif
#ifndef HAS_I64
#define HAS_I64 0
#endif
#ifndef HAS_U64
#define HAS_U64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { ArgminAxisPush push; }

#if HAS_I64
#define ARGMIN_FLOAT(NAME, LOAD_FN) \
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
        float best = 0.0f; \
        int64_t best_idx = 0; \
        linear_to_coords(out_idx, out_desc, out_coords); \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                float v = LOAD_FN(input_desc, i, data); \
                int64_t axis_idx = int64_t(in_coords[push.axis]); \
                bool better = push.select_first != 0u ? (v < best) : (v <= best); \
                if (!initialized) { \
                    best = v; \
                    best_idx = axis_idx; \
                    initialized = true; \
                } else if (better) { \
                    best = v; \
                    best_idx = axis_idx; \
                } \
            } \
        } \
        store_i64(out_desc, out_idx, best_idx, out_buf); \
    }

#define ARGMIN_INT(NAME, LOAD_FN) \
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
                int v = int(LOAD_FN(input_desc, i, data)); \
                int64_t axis_idx = int64_t(in_coords[push.axis]); \
                bool better = push.select_first != 0u ? (v < best) : (v <= best); \
                if (!initialized) { \
                    best = v; \
                    best_idx = axis_idx; \
                    initialized = true; \
                } else if (better) { \
                    best = v; \
                    best_idx = axis_idx; \
                } \
            } \
        } \
        store_i64(out_desc, out_idx, best_idx, out_buf); \
    }

#define ARGMIN_UINT(NAME, LOAD_FN) \
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
                uint v = uint(LOAD_FN(input_desc, i, data)); \
                int64_t axis_idx = int64_t(in_coords[push.axis]); \
                bool better = push.select_first != 0u ? (v < best) : (v <= best); \
                if (!initialized) { \
                    best = v; \
                    best_idx = axis_idx; \
                    initialized = true; \
                } else if (better) { \
                    best = v; \
                    best_idx = axis_idx; \
                } \
            } \
        } \
        store_i64(out_desc, out_idx, best_idx, out_buf); \
    }

ARGMIN_FLOAT(argmin_axis_f8_normal, load_f8)
ARGMIN_FLOAT(argmin_axis_f16_normal, load_f16)
ARGMIN_FLOAT(argmin_axis_bf16_normal, load_bf16)
ARGMIN_FLOAT(argmin_axis_f32_normal, load_f32)
#if HAS_F64
ARGMIN_FLOAT(argmin_axis_f64_normal, load_f64)
#endif

ARGMIN_INT(argmin_axis_i8_normal, load_i8)
ARGMIN_INT(argmin_axis_i16_normal, load_i16)
ARGMIN_INT(argmin_axis_i32_normal, load_i32)
ARGMIN_INT(argmin_axis_i64_normal, load_i64)

ARGMIN_UINT(argmin_axis_u8_normal, load_u8_val)
ARGMIN_UINT(argmin_axis_u16_normal, load_u16_val)
ARGMIN_UINT(argmin_axis_u32_normal, load_u32)
#if HAS_U64
ARGMIN_UINT(argmin_axis_u64_normal, load_u64)
#endif

#undef ARGMIN_FLOAT
#undef ARGMIN_INT
#undef ARGMIN_UINT
#endif
