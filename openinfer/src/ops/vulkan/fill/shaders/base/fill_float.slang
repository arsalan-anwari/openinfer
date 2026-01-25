struct PushConsts {
    uint len;
    float value_f32;
    double value_f64;
};

[[vk::push_constant]] PushConsts pc;

struct FillBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<FillBuffers<half>> gFill_f16_native;
ParameterBlock<FillBuffers<float>> gFill_f32;
ParameterBlock<FillBuffers<double>> gFill_f64;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_f16_native.output0[i] = (half)pc.value_f32;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_f32.output0[i] = pc.value_f32;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_f64.output0[i] = pc.value_f64;
    }
}
