#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
#include "common.slang"

#ifndef HAS_I64
#define HAS_I64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { AbsPush push; }

// ACC = I8
[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i1_accumulate_i8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i8(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i2_accumulate_i8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i8(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i4_accumulate_i8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i8(tensor_descs[1], idx, acc, out_buf);
}

// ACC = I16
[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i1_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i16(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i2_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i16(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i4_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i16(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i8_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_i8(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i16(tensor_descs[1], idx, acc, out_buf);
}

// ACC = I32
[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i1_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i32(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i2_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i32(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i4_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i32(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i8_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_i8(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i32(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i16_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_i16(tensor_descs[0], idx, data);
    int acc = (value < 0) ? -value : value;
    store_i32(tensor_descs[1], idx, acc, out_buf);
}

// ACC = I64
#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i1_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int64_t acc = (value < 0) ? -int64_t(value) : int64_t(value);
    store_i64(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i2_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int64_t acc = (value < 0) ? -int64_t(value) : int64_t(value);
    store_i64(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i4_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_byte_i(tensor_descs[0], idx, data);
    int64_t acc = (value < 0) ? -int64_t(value) : int64_t(value);
    store_i64(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i8_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_i8(tensor_descs[0], idx, data);
    int64_t acc = (value < 0) ? -int64_t(value) : int64_t(value);
    store_i64(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i16_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_i16(tensor_descs[0], idx, data);
    int64_t acc = (value < 0) ? -int64_t(value) : int64_t(value);
    store_i64(tensor_descs[1], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i32_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_i32(tensor_descs[0], idx, data);
    int64_t acc = (value < 0) ? -int64_t(value) : int64_t(value);
    store_i64(tensor_descs[1], idx, acc, out_buf);
}
#endif
