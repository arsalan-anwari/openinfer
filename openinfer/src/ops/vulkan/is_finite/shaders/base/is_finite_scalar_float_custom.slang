#include "../../../float_utils.slang"

struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct IsFiniteScalarBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<IsFiniteScalarBuffers> gIsFinite_f8;
ParameterBlock<IsFiniteScalarBuffers> gIsFinite_f16;
ParameterBlock<IsFiniteScalarBuffers> gIsFinite_bf16;

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = read_f8(gIsFinite_f8.input0, i);
        float abs_x = abs(x);
        if (x != x || abs_x > (float)3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f8.output0[0] = ok;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = read_f16(gIsFinite_f16.input0, i);
        float abs_x = abs(x);
        if (x != x || abs_x > (float)3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f16.output0[0] = ok;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_bf16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = read_bf16(gIsFinite_bf16.input0, i);
        float abs_x = abs(x);
        if (x != x || abs_x > (float)3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_bf16.output0[0] = ok;
}
