#include "../../../float_utils.slang"

struct PushConsts {
    uint len;
    uint value_bits;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct FillBytesBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<FillBytesBuffers> gFill_f8;
ParameterBlock<FillBytesBuffers> gFill_f16;
ParameterBlock<FillBytesBuffers> gFill_bf16;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        write_f8(gFill_f8.output0, i, value);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        write_f16(gFill_f16.output0, i, value);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_bf16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        write_bf16(gFill_bf16.output0, i, value);
    }
}
