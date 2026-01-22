#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddAccBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t, Std430DataLayout> output0;
};

ParameterBlock<AddAccBuffers> gAddAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u1_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a = packed_read_unsigned(gAddAcc.input0, i, 1);
    uint b = packed_read_unsigned(gAddAcc.input1, i, 1);
    gAddAcc.output0[i] = (uint64_t)(uint)(a + b);
}
}
