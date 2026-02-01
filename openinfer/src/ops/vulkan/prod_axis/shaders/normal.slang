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
[[vk::push_constant]] cbuffer Push { ProdAxisPush push; }

#define PROD_AXIS_FLOAT(NAME, LOAD_FN, STORE_FN) \
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
        float acc = 1.0f; \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                acc *= LOAD_FN(input_desc, i, data); \
            } \
        } \
        STORE_FN(out_desc, out_idx, acc, out_buf); \
    }

#define PROD_AXIS_INT(NAME, LOAD_FN, STORE_FN) \
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
        int acc = 1; \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                acc *= int(LOAD_FN(input_desc, i, data)); \
            } \
        } \
        STORE_FN(out_desc, out_idx, acc, out_buf); \
    }

#define PROD_AXIS_UINT(NAME, LOAD_FN, STORE_FN) \
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
        uint acc = 1u; \
        for (uint i = 0u; i < push.input_len; ++i) { \
            linear_to_coords(i, input_desc, in_coords); \
            if (coords_match(input_desc.rank, out_desc.rank, push.axes_mask, push.keepdims, in_coords, out_coords)) { \
                acc *= uint(LOAD_FN(input_desc, i, data)); \
            } \
        } \
        STORE_FN(out_desc, out_idx, acc, out_buf); \
    }

PROD_AXIS_FLOAT(prod_axis_f8_normal, load_f8, store_f8)
PROD_AXIS_FLOAT(prod_axis_f16_normal, load_f16, store_f16)
PROD_AXIS_FLOAT(prod_axis_bf16_normal, load_bf16, store_bf16)
PROD_AXIS_FLOAT(prod_axis_f32_normal, load_f32, store_f32)
#if HAS_F64
PROD_AXIS_FLOAT(prod_axis_f64_normal, load_f64, store_f64)
#endif

PROD_AXIS_INT(prod_axis_i8_normal, load_i8, store_i8)
PROD_AXIS_INT(prod_axis_i16_normal, load_i16, store_i16)
PROD_AXIS_INT(prod_axis_i32_normal, load_i32, store_i32)
#if HAS_I64
PROD_AXIS_INT(prod_axis_i64_normal, load_i64, store_i64)
#endif

PROD_AXIS_UINT(prod_axis_u8_normal, load_u8_val, store_u8_val)
PROD_AXIS_UINT(prod_axis_u16_normal, load_u16_val, store_u16_val)
PROD_AXIS_UINT(prod_axis_u32_normal, load_u32, store_u32)
#if HAS_U64
PROD_AXIS_UINT(prod_axis_u64_normal, load_u64, store_u64)
#endif

#undef PROD_AXIS_FLOAT
#undef PROD_AXIS_INT
#undef PROD_AXIS_UINT
