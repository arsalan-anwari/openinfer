// Relu inplace kernel (signed integers).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct ReluBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<ReluBuffers<int>> gRelu_i8;
ParameterBlock<ReluBuffers<int>> gRelu_i16;
ParameterBlock<ReluBuffers<int>> gRelu_i32;
ParameterBlock<ReluBuffers<int64_t>> gRelu_i64;

float reluFloat(float x, float negative_slope, float clamp_max) {
    float y = (x >= 0.0f) ? x : (x * negative_slope);
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_inplace_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float negative_slope = asfloat(pc.flags);
        float clamp_max = asfloat(pc.pad0);
        float y = reluFloat((float)gRelu_i8.input0[i], negative_slope, clamp_max);
        y = clamp(y, -128.0f, 127.0f);
        gRelu_i8.output0[i] = (int)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_inplace_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float negative_slope = asfloat(pc.flags);
        float clamp_max = asfloat(pc.pad0);
        float y = reluFloat((float)gRelu_i16.input0[i], negative_slope, clamp_max);
        y = clamp(y, -32768.0f, 32767.0f);
        gRelu_i16.output0[i] = (int)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_inplace_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float negative_slope = asfloat(pc.flags);
        float clamp_max = asfloat(pc.pad0);
        float y = reluFloat((float)gRelu_i32.input0[i], negative_slope, clamp_max);
        y = clamp(y, -2147483648.0f, 2147483647.0f);
        gRelu_i32.output0[i] = (int)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_inplace_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float negative_slope = asfloat(pc.flags);
        float clamp_max = asfloat(pc.pad0);
        float y = reluFloat((float)gRelu_i64.input0[i], negative_slope, clamp_max);
        gRelu_i64.output0[i] = (int64_t)y;
    }
}
