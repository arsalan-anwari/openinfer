#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AbsBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<AbsBuffers<int>> gAbs_i8;
ParameterBlock<AbsBuffers<int>> gAbs_i16;
ParameterBlock<AbsBuffers<int>> gAbs_i32;
ParameterBlock<AbsBuffers<int64_t>> gAbs_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gAbs_i8.input0[i];
        gAbs_i8.output0[i] = x < 0 ? -x : x;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gAbs_i16.input0[i];
        gAbs_i16.output0[i] = x < 0 ? -x : x;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = gAbs_i32.input0[i];
        gAbs_i32.output0[i] = x < 0 ? -x : x;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int64_t x = gAbs_i64.input0[i];
        gAbs_i64.output0[i] = x < 0 ? -x : x;
    }
}
