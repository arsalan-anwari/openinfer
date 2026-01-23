#include "../../../float_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddBytesBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<AddBytesBuffers> gAdd_f8;
ParameterBlock<AddBytesBuffers> gAdd_f16;
ParameterBlock<AddBytesBuffers> gAdd_bf16;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_inplace_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAdd_f8.meta0);
        uint b_idx = broadcastOffset(i, gAdd_f8.meta1);
        float a = read_f8(gAdd_f8.input0, a_idx);
        float b = read_f8(gAdd_f8.input1, b_idx);
        write_f8(gAdd_f8.output0, i, a + b);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_inplace_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAdd_f16.meta0);
        uint b_idx = broadcastOffset(i, gAdd_f16.meta1);
        float a = read_f16(gAdd_f16.input0, a_idx);
        float b = read_f16(gAdd_f16.input1, b_idx);
        write_f16(gAdd_f16.output0, i, a + b);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_inplace_bf16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAdd_bf16.meta0);
        uint b_idx = broadcastOffset(i, gAdd_bf16.meta1);
        float a = read_bf16(gAdd_bf16.input0, a_idx);
        float b = read_bf16(gAdd_bf16.input1, b_idx);
        write_bf16(gAdd_bf16.output0, i, a + b);
    }
}
