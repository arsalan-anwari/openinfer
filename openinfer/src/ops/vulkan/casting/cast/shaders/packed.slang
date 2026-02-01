#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
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

static const uint DTYPE_I8 = 1u;
static const uint DTYPE_I16 = 2u;
static const uint DTYPE_I32 = 3u;
static const uint DTYPE_I64 = 4u;
static const uint DTYPE_U8 = 5u;
static const uint DTYPE_U16 = 6u;
static const uint DTYPE_U32 = 7u;
static const uint DTYPE_U64 = 8u;
static const uint DTYPE_F16 = 9u;
static const uint DTYPE_BF16 = 10u;
static const uint DTYPE_F32 = 11u;
static const uint DTYPE_F64 = 12u;
static const uint DTYPE_F8 = 13u;

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { CastPush push; }

int clamp_i8(double v) {
    if (isnan(v)) return 0;
    if (v < -128.0) return -128;
    if (v > 127.0) return 127;
    return (int)v;
}
int clamp_i16(double v) {
    if (isnan(v)) return 0;
    if (v < -32768.0) return -32768;
    if (v > 32767.0) return 32767;
    return (int)v;
}
int clamp_i32(double v) {
    if (isnan(v)) return 0;
    if (v < -2147483648.0) return -2147483648;
    if (v > 2147483647.0) return 2147483647;
    return (int)v;
}
#if HAS_I64
int64_t clamp_i64(double v) {
    if (isnan(v)) return 0;
    if (v < -9.223372036854776e18) return int64_t(-9223372036854775807LL - 1LL);
    if (v > 9.223372036854776e18) return int64_t(9223372036854775807LL);
    return int64_t(v);
}
#endif
uint clamp_u8(double v) {
    if (isnan(v) || v < 0.0) return 0u;
    if (v > 255.0) return 255u;
    return (uint)v;
}
uint clamp_u16(double v) {
    if (isnan(v) || v < 0.0) return 0u;
    if (v > 65535.0) return 65535u;
    return (uint)v;
}
uint clamp_u32(double v) {
    if (isnan(v) || v < 0.0) return 0u;
    if (v > 4294967295.0) return 4294967295u;
    return (uint)v;
}
#if HAS_U64
uint64_t clamp_u64(double v) {
    if (isnan(v) || v < 0.0) return uint64_t(0);
    if (v > 1.8446744073709552e19) return uint64_t(18446744073709551615ull);
    return uint64_t(v);
}
#endif

void store_cast(TensorDesc out_desc, uint idx, double v) {
    uint dtype = out_desc.dtype;
    if (dtype == DTYPE_F8) {
        store_f8(out_desc, idx, (float)v, out_buf);
    } else if (dtype == DTYPE_F16) {
        store_f16(out_desc, idx, (float)v, out_buf);
    } else if (dtype == DTYPE_BF16) {
        store_bf16(out_desc, idx, (float)v, out_buf);
    } else if (dtype == DTYPE_F32) {
        store_f32(out_desc, idx, (float)v, out_buf);
    } else if (dtype == DTYPE_F64) {
#if HAS_F64
        store_f64(out_desc, idx, v, out_buf);
#endif
    } else if (dtype == DTYPE_I8) {
        store_i8(out_desc, idx, clamp_i8(v), out_buf);
    } else if (dtype == DTYPE_I16) {
        store_i16(out_desc, idx, clamp_i16(v), out_buf);
    } else if (dtype == DTYPE_I32) {
        store_i32(out_desc, idx, clamp_i32(v), out_buf);
    } else if (dtype == DTYPE_I64) {
#if HAS_I64
        store_i64(out_desc, idx, clamp_i64(v), out_buf);
#endif
    } else if (dtype == DTYPE_U8) {
        store_u8_val(out_desc, idx, clamp_u8(v), out_buf);
    } else if (dtype == DTYPE_U16) {
        store_u16_val(out_desc, idx, clamp_u16(v), out_buf);
    } else if (dtype == DTYPE_U32) {
        store_u32(out_desc, idx, clamp_u32(v), out_buf);
    } else if (dtype == DTYPE_U64) {
#if HAS_U64
        store_u64(out_desc, idx, clamp_u64(v), out_buf);
#endif
    }
}

#define CAST_PACKED_ENTRY(NAME, LOAD_FN) \
    [numthreads(256, 1, 1)] \
    [shader("compute")] \
    void NAME(uint3 tid : SV_DispatchThreadID) { \
        uint idx = tid.x; \
        if (idx >= push.len) return; \
        TensorDesc in_desc = tensor_descs[0]; \
        TensorDesc out_desc = tensor_descs[1]; \
        double v = LOAD_FN; \
        store_cast(out_desc, idx, v); \
    }

CAST_PACKED_ENTRY(cast_i1_packed, (double)load_packed_i(in_desc, idx, data))
CAST_PACKED_ENTRY(cast_i2_packed, (double)load_packed_i(in_desc, idx, data))
CAST_PACKED_ENTRY(cast_i4_packed, (double)load_packed_i(in_desc, idx, data))
CAST_PACKED_ENTRY(cast_u1_packed, (double)load_packed_u(in_desc, idx, data))
CAST_PACKED_ENTRY(cast_u2_packed, (double)load_packed_u(in_desc, idx, data))
CAST_PACKED_ENTRY(cast_u4_packed, (double)load_packed_u(in_desc, idx, data))

#undef CAST_PACKED_ENTRY
