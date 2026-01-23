#include "../../../float_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AbsBytesBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<AbsBytesBuffers> gAbs_f8;
ParameterBlock<AbsBytesBuffers> gAbs_f16;
ParameterBlock<AbsBytesBuffers> gAbs_bf16;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = read_f8(gAbs_f8.input0, i);
        write_f8(gAbs_f8.output0, i, abs(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = read_f16(gAbs_f16.input0, i);
        write_f16(gAbs_f16.output0, i, abs(x));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_bf16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = read_bf16(gAbs_bf16.input0, i);
        write_bf16(gAbs_bf16.output0, i, abs(x));
    }
}
