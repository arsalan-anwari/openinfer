#include "../../../shaders/common.slang"
#include "../../../shaders/float_utils.slang"
#include "common.slang"



#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { IsFinitePush push; }

void write_result(uint ok) {
    TensorDesc out = tensor_descs[1];
    store_bool_at(out, 0, ok, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_finite_f32_normal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x != 0) return;
    TensorDesc a = tensor_descs[0];
    uint ok = 1u;
    for (uint i = 0; i < push.input_len; i++) {
        float v = load_f32(a, i, data);
        if (isnan(v) || isinf(v)) {
            ok = 0u;
            break;
        }
    }
    write_result(ok);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_finite_f16_normal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x != 0) return;
    TensorDesc a = tensor_descs[0];
    uint ok = 1u;
    for (uint i = 0; i < push.input_len; i++) {
        float v = load_f16(a, i, data);
        if (isnan(v) || isinf(v)) {
            ok = 0u;
            break;
        }
    }
    write_result(ok);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_finite_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x != 0) return;
    TensorDesc a = tensor_descs[0];
    uint ok = 1u;
    for (uint i = 0; i < push.input_len; i++) {
        float v = load_bf16(a, i, data);
        if (isnan(v) || isinf(v)) {
            ok = 0u;
            break;
        }
    }
    write_result(ok);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_finite_f8_normal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x != 0) return;
    TensorDesc a = tensor_descs[0];
    uint ok = 1u;
    for (uint i = 0; i < push.input_len; i++) {
        float v = load_f8(a, i, data);
        if (isnan(v) || isinf(v)) {
            ok = 0u;
            break;
        }
    }
    write_result(ok);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void is_finite_f64_normal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x != 0) return;
    TensorDesc a = tensor_descs[0];
    uint ok = 1u;
    for (uint i = 0; i < push.input_len; i++) {
        double v = load_f64(a, i, data);
        if (isnan(v) || isinf(v)) {
            ok = 0u;
            break;
        }
    }
    write_result(ok);
}
#endif
