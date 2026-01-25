struct PushConsts {
    uint len;
    float alpha_f32;
    float clamp_f32;
    double alpha_f64;
    double clamp_f64;
};

[[vk::push_constant]] PushConsts pc;

float relu_float(float x, float alpha, float clamp_max) {
    float y = x >= 0.0f ? x : x * alpha;
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}

struct ReluBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<ReluBuffers<int>> gRelu_i8;
ParameterBlock<ReluBuffers<int>> gRelu_i16;
ParameterBlock<ReluBuffers<int>> gRelu_i32;
ParameterBlock<ReluBuffers<int64_t>> gRelu_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gRelu_i8.input0[i];
        float y = relu_float((float)x, pc.alpha_f32, pc.clamp_f32);
        if (y < -128.0f) {
            y = -128.0f;
        }
        if (y > 127.0f) {
            y = 127.0f;
        }
        gRelu_i8.output0[i] = (int)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gRelu_i16.input0[i];
        float y = relu_float((float)x, pc.alpha_f32, pc.clamp_f32);
        if (y < -32768.0f) {
            y = -32768.0f;
        }
        if (y > 32767.0f) {
            y = 32767.0f;
        }
        gRelu_i16.output0[i] = (int)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gRelu_i32.input0[i];
        float y = relu_float((float)x, pc.alpha_f32, pc.clamp_f32);
        if (y < -2147483648.0f) {
            y = -2147483648.0f;
        }
        if (y > 2147483647.0f) {
            y = 2147483647.0f;
        }
        gRelu_i32.output0[i] = (int)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int64_t x = gRelu_i64.input0[i];
        float y = relu_float((float)x, pc.alpha_f32, pc.clamp_f32);
        if (y < -9223372036854775808.0f) {
            y = -9223372036854775808.0f;
        }
        if (y > 9223372036854775807.0f) {
            y = 9223372036854775807.0f;
        }
        gRelu_i64.output0[i] = (int64_t)y;
    }
}
