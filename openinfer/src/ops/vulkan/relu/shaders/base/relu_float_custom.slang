#include "../../../float_utils.slang"

struct PushConsts {
    uint len;
    uint alpha_lo;
    uint alpha_hi;
    uint clamp_lo;
    uint clamp_hi;
    uint flags;
};

[[vk::push_constant]] PushConsts pc;

float relu_float(float x, float alpha, float clamp_max) {
    float y = x >= 0.0f ? x : x * alpha;
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}

struct ReluBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<ReluBuffers> gRelu_f8;
ParameterBlock<ReluBuffers> gRelu_f16;
ParameterBlock<ReluBuffers> gRelu_bf16;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = read_f8(gRelu_f8.input0, i);
        float y = relu_float(x, asfloat(pc.alpha_lo), asfloat(pc.clamp_lo));
        write_f8(gRelu_f8.output0, i, y);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = read_f16(gRelu_f16.input0, i);
        float y = relu_float(x, asfloat(pc.alpha_lo), asfloat(pc.clamp_lo));
        write_f16(gRelu_f16.output0, i, y);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_bf16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = read_bf16(gRelu_bf16.input0, i);
        float y = relu_float(x, asfloat(pc.alpha_lo), asfloat(pc.clamp_lo));
        write_bf16(gRelu_bf16.output0, i, y);
    }
}
