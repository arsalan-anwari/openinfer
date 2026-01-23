#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AbsAccBuffersInt {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

struct AbsAccBuffersInt64 {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
};

ParameterBlock<AbsAccBuffersInt> gAbsAcc_i16_i32;
ParameterBlock<AbsAccBuffersInt64> gAbsAcc_i16_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i16_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gAbsAcc_i16_i32.input0[i];
        int y = x < 0 ? -x : x;
        gAbsAcc_i16_i32.output0[i] = y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i16_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int64_t x = (int64_t)gAbsAcc_i16_i64.input0[i];
        int64_t y = x < 0 ? -x : x;
        gAbsAcc_i16_i64.output0[i] = y;
    }
}
