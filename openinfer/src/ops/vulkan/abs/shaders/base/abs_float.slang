#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AbsBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<AbsBuffers<half>> gAbs_f16_native;
ParameterBlock<AbsBuffers<float>> gAbs_f32;
ParameterBlock<AbsBuffers<double>> gAbs_f64;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        half x = gAbs_f16_native.input0[i];
        gAbs_f16_native.output0[i] = abs(x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = gAbs_f32.input0[i];
        gAbs_f32.output0[i] = abs(x);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        double x = gAbs_f64.input0[i];
        gAbs_f64.output0[i] = abs(x);
    }
}
