#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
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
[[vk::push_constant]] cbuffer Push { AddPush push; }

// ACC = I8
[numthreads(256, 1, 1)]
[shader("compute")]
void add_i1_accumulate_i8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i8(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i2_accumulate_i8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i8(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i4_accumulate_i8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i8(tensor_descs[2], idx, acc, out_buf);
}

// ACC = I16
[numthreads(256, 1, 1)]
[shader("compute")]
void add_i1_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i16(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i2_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i16(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i4_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i16(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i8_accumulate_i16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_i8(tensor_descs[0], idx, data))
        + int(load_i8(tensor_descs[1], idx, data));
    store_i16(tensor_descs[2], idx, acc, out_buf);
}

// ACC = I32
[numthreads(256, 1, 1)]
[shader("compute")]
void add_i1_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i2_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i4_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_packed_i(tensor_descs[0], idx, data))
        + int(load_packed_i(tensor_descs[1], idx, data));
    store_i32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i8_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_i8(tensor_descs[0], idx, data))
        + int(load_i8(tensor_descs[1], idx, data));
    store_i32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i16_accumulate_i32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int acc = int(load_i16(tensor_descs[0], idx, data))
        + int(load_i16(tensor_descs[1], idx, data));
    store_i32(tensor_descs[2], idx, acc, out_buf);
}

// ACC = I64
#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void add_i1_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int64_t acc = int64_t(load_packed_i(tensor_descs[0], idx, data))
        + int64_t(load_packed_i(tensor_descs[1], idx, data));
    store_i64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i2_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int64_t acc = int64_t(load_packed_i(tensor_descs[0], idx, data))
        + int64_t(load_packed_i(tensor_descs[1], idx, data));
    store_i64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i4_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int64_t acc = int64_t(load_packed_i(tensor_descs[0], idx, data))
        + int64_t(load_packed_i(tensor_descs[1], idx, data));
    store_i64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i8_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int64_t acc = int64_t(load_i8(tensor_descs[0], idx, data))
        + int64_t(load_i8(tensor_descs[1], idx, data));
    store_i64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i16_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int64_t acc = int64_t(load_i16(tensor_descs[0], idx, data))
        + int64_t(load_i16(tensor_descs[1], idx, data));
    store_i64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i32_accumulate_i64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int64_t acc = int64_t(load_i32(tensor_descs[0], idx, data))
        + int64_t(load_i32(tensor_descs[1], idx, data));
    store_i64(tensor_descs[2], idx, acc, out_buf);
}
#endif

// ACC = U8
[numthreads(256, 1, 1)]
[shader("compute")]
void add_u1_accumulate_u8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u8_val(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u2_accumulate_u8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u8_val(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u4_accumulate_u8(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u8_val(tensor_descs[2], idx, acc, out_buf);
}

// ACC = U16
[numthreads(256, 1, 1)]
[shader("compute")]
void add_u1_accumulate_u16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u16_val(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u2_accumulate_u16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u16_val(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u4_accumulate_u16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u16_val(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u8_accumulate_u16(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_u8_val(tensor_descs[0], idx, data)
        + load_u8_val(tensor_descs[1], idx, data);
    store_u16_val(tensor_descs[2], idx, acc, out_buf);
}

// ACC = U32
[numthreads(256, 1, 1)]
[shader("compute")]
void add_u1_accumulate_u32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u2_accumulate_u32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u4_accumulate_u32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_packed_u(tensor_descs[0], idx, data)
        + load_packed_u(tensor_descs[1], idx, data);
    store_u32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u8_accumulate_u32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_u8_val(tensor_descs[0], idx, data)
        + load_u8_val(tensor_descs[1], idx, data);
    store_u32(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u16_accumulate_u32(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint acc = load_u16_val(tensor_descs[0], idx, data)
        + load_u16_val(tensor_descs[1], idx, data);
    store_u32(tensor_descs[2], idx, acc, out_buf);
}

// ACC = U64 
#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void add_u1_accumulate_u64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint64_t acc = uint64_t(load_packed_u(tensor_descs[0], idx, data))
        + uint64_t(load_packed_u(tensor_descs[1], idx, data));
    store_u64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u2_accumulate_u64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint64_t acc = uint64_t(load_packed_u(tensor_descs[0], idx, data))
        + uint64_t(load_packed_u(tensor_descs[1], idx, data));
    store_u64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u4_accumulate_u64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint64_t acc = uint64_t(load_packed_u(tensor_descs[0], idx, data))
        + uint64_t(load_packed_u(tensor_descs[1], idx, data));
    store_u64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u8_accumulate_u64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint64_t acc = uint64_t(load_u8_val(tensor_descs[0], idx, data))
        + uint64_t(load_u8_val(tensor_descs[1], idx, data));
    store_u64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u16_accumulate_u64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint64_t acc = uint64_t(load_u16_val(tensor_descs[0], idx, data))
        + uint64_t(load_u16_val(tensor_descs[1], idx, data));
    store_u64(tensor_descs[2], idx, acc, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u32_accumulate_u64(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint64_t acc = uint64_t(load_u32(tensor_descs[0], idx, data))
        + uint64_t(load_u32(tensor_descs[1], idx, data));
    store_u64(tensor_descs[2], idx, acc, out_buf);
}
#endif
