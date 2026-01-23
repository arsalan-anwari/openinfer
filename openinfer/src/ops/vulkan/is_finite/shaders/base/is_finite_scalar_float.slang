struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct IsFiniteScalarBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<IsFiniteScalarBuffers<half>> gIsFinite_f16_native;
ParameterBlock<IsFiniteScalarBuffers<float>> gIsFinite_f32;
ParameterBlock<IsFiniteScalarBuffers<double>> gIsFinite_f64;

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = (float)gIsFinite_f16_native.input0[i];
        float abs_x = abs(x);
        if (x != x || abs_x > (float)3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f16_native.output0[0] = ok;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = gIsFinite_f32.input0[i];
        float abs_x = abs(x);
        if (x != x || abs_x > (float)3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f32.output0[0] = ok;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        double x = gIsFinite_f64.input0[i];
        double abs_x = abs(x);
        if (x != x || abs_x > (double)1.7976931348623157e308) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f64.output0[0] = ok;
}
