#define OI_BROADCAST

#include "../../../float_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulBytesBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
#ifdef OI_BROADCAST
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<MulBytesBuffers> gMul_f8;
ParameterBlock<MulBytesBuffers> gMul_f16;
ParameterBlock<MulBytesBuffers> gMul_bf16;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_f8_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMul_f8.meta0);
        b_idx = broadcastOffset(i, gMul_f8.meta1);
#endif
        float a = read_f8(gMul_f8.input0, a_idx);
        float b = read_f8(gMul_f8.input1, b_idx);
        write_f8(gMul_f8.output0, i, a * b);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_f16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMul_f16.meta0);
        b_idx = broadcastOffset(i, gMul_f16.meta1);
#endif
        float a = read_f16(gMul_f16.input0, a_idx);
        float b = read_f16(gMul_f16.input1, b_idx);
        write_f16(gMul_f16.output0, i, a * b);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_bf16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMul_bf16.meta0);
        b_idx = broadcastOffset(i, gMul_bf16.meta1);
#endif
        float a = read_bf16(gMul_bf16.input0, a_idx);
        float b = read_bf16(gMul_bf16.input1, b_idx);
        write_bf16(gMul_bf16.output0, i, a * b);
    }
}
