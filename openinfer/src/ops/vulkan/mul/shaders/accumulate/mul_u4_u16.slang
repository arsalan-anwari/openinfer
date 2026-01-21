#include "../../../packed/packed_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulAccBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<MulAccBuffers> gMulAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_u4_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a = packed_read_unsigned(gMulAcc.input0, i, 4);
    uint b = packed_read_unsigned(gMulAcc.input1, i, 4);
    gMulAcc.output0[i] = (uint)(uint)(a * b);
}
}
