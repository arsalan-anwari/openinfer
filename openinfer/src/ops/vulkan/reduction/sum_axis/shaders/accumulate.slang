#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
#include "../../../shaders/reduce_utils.slang"
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
[[vk::push_constant]] cbuffer Push { SumAxisPush push; }

#define SUM_AXIS_ACC_SIGNED(NAME, LOAD_FN, STORE_FN) \
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
                acc += LOAD_FN; \
            } \
        } \
        STORE_FN(out_desc, out_idx, acc, out_buf); \
    }

#define SUM_AXIS_ACC_UNSIGNED(NAME, LOAD_FN, STORE_FN) \
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
                acc += LOAD_FN; \
            } \
        } \
        STORE_FN(out_desc, out_idx, acc, out_buf); \
    }

// Packed signed inputs.
SUM_AXIS_ACC_SIGNED(sum_axis_i1_accumulate_i8, load_packed_i(input_desc, i, data), store_i8)
SUM_AXIS_ACC_SIGNED(sum_axis_i1_accumulate_i16, load_packed_i(input_desc, i, data), store_i16)
SUM_AXIS_ACC_SIGNED(sum_axis_i1_accumulate_i32, load_packed_i(input_desc, i, data), store_i32)
#if HAS_I64
SUM_AXIS_ACC_SIGNED(sum_axis_i1_accumulate_i64, load_packed_i(input_desc, i, data), store_i64)
#endif
SUM_AXIS_ACC_SIGNED(sum_axis_i2_accumulate_i8, load_packed_i(input_desc, i, data), store_i8)
SUM_AXIS_ACC_SIGNED(sum_axis_i2_accumulate_i16, load_packed_i(input_desc, i, data), store_i16)
SUM_AXIS_ACC_SIGNED(sum_axis_i2_accumulate_i32, load_packed_i(input_desc, i, data), store_i32)
#if HAS_I64
SUM_AXIS_ACC_SIGNED(sum_axis_i2_accumulate_i64, load_packed_i(input_desc, i, data), store_i64)
#endif
SUM_AXIS_ACC_SIGNED(sum_axis_i4_accumulate_i8, load_packed_i(input_desc, i, data), store_i8)
SUM_AXIS_ACC_SIGNED(sum_axis_i4_accumulate_i16, load_packed_i(input_desc, i, data), store_i16)
SUM_AXIS_ACC_SIGNED(sum_axis_i4_accumulate_i32, load_packed_i(input_desc, i, data), store_i32)
#if HAS_I64
SUM_AXIS_ACC_SIGNED(sum_axis_i4_accumulate_i64, load_packed_i(input_desc, i, data), store_i64)
#endif

SUM_AXIS_ACC_SIGNED(sum_axis_i8_accumulate_i16, load_i8(input_desc, i, data), store_i16)
SUM_AXIS_ACC_SIGNED(sum_axis_i8_accumulate_i32, load_i8(input_desc, i, data), store_i32)
#if HAS_I64
SUM_AXIS_ACC_SIGNED(sum_axis_i8_accumulate_i64, load_i8(input_desc, i, data), store_i64)
SUM_AXIS_ACC_SIGNED(sum_axis_i16_accumulate_i64, load_i16(input_desc, i, data), store_i64)
SUM_AXIS_ACC_SIGNED(sum_axis_i32_accumulate_i64, load_i32(input_desc, i, data), store_i64)
#endif
SUM_AXIS_ACC_SIGNED(sum_axis_i16_accumulate_i32, load_i16(input_desc, i, data), store_i32)

// Packed unsigned inputs.
SUM_AXIS_ACC_UNSIGNED(sum_axis_u1_accumulate_u8, load_packed_u(input_desc, i, data), store_u8_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u1_accumulate_u16, load_packed_u(input_desc, i, data), store_u16_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u1_accumulate_u32, load_packed_u(input_desc, i, data), store_u32)
#if HAS_U64
SUM_AXIS_ACC_UNSIGNED(sum_axis_u1_accumulate_u64, load_packed_u(input_desc, i, data), store_u64)
#endif
SUM_AXIS_ACC_UNSIGNED(sum_axis_u2_accumulate_u8, load_packed_u(input_desc, i, data), store_u8_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u2_accumulate_u16, load_packed_u(input_desc, i, data), store_u16_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u2_accumulate_u32, load_packed_u(input_desc, i, data), store_u32)
#if HAS_U64
SUM_AXIS_ACC_UNSIGNED(sum_axis_u2_accumulate_u64, load_packed_u(input_desc, i, data), store_u64)
#endif
SUM_AXIS_ACC_UNSIGNED(sum_axis_u4_accumulate_u8, load_packed_u(input_desc, i, data), store_u8_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u4_accumulate_u16, load_packed_u(input_desc, i, data), store_u16_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u4_accumulate_u32, load_packed_u(input_desc, i, data), store_u32)
#if HAS_U64
SUM_AXIS_ACC_UNSIGNED(sum_axis_u4_accumulate_u64, load_packed_u(input_desc, i, data), store_u64)
#endif

SUM_AXIS_ACC_UNSIGNED(sum_axis_u8_accumulate_u16, load_u8_val(input_desc, i, data), store_u16_val)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u8_accumulate_u32, load_u8_val(input_desc, i, data), store_u32)
#if HAS_U64
SUM_AXIS_ACC_UNSIGNED(sum_axis_u8_accumulate_u64, load_u8_val(input_desc, i, data), store_u64)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u16_accumulate_u64, load_u16_val(input_desc, i, data), store_u64)
SUM_AXIS_ACC_UNSIGNED(sum_axis_u32_accumulate_u64, load_u32(input_desc, i, data), store_u64)
#endif
SUM_AXIS_ACC_UNSIGNED(sum_axis_u16_accumulate_u32, load_u16_val(input_desc, i, data), store_u32)

#undef SUM_AXIS_ACC_SIGNED
#undef SUM_AXIS_ACC_UNSIGNED
