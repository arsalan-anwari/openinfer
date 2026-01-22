#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsAccBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

ParameterBlock<AbsAccBuffers> gAbsAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i4_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int x = packed_read_signed(gAbsAcc.input0, i, 4);
    int y = x < 0 ? -x : x;
    gAbsAcc.output0[i] = (int)y;
}
}
