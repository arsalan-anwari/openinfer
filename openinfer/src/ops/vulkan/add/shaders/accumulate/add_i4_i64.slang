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
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
};

ParameterBlock<AddAccBuffers> gAddAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i4_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int a = packed_read_signed(gAddAcc.input0, i, 4);
    int b = packed_read_signed(gAddAcc.input1, i, 4);
    gAddAcc.output0[i] = (int64_t)(int)(a + b);
}
}
