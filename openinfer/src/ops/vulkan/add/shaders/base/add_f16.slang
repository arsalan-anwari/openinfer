#include "../../../float_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<AddBuffers> gAdd;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float a = read_f16(gAdd.input0, i);
    float b = read_f16(gAdd.input1, i);
    float y = a + b;
    write_f16(gAdd.output0, i, y);
}
}
