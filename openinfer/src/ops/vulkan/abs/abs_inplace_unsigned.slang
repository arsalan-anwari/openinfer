// Abs inplace kernel (unsigned integers and bool).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<AbsBuffers<uint>> gAbs_u8;
ParameterBlock<AbsBuffers<uint>> gAbs_u16;
ParameterBlock<AbsBuffers<uint>> gAbs_u32;
ParameterBlock<AbsBuffers<uint64_t>> gAbs_u64;
ParameterBlock<AbsBuffers<uint>> gAbs_bool;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gAbs_u8.output0[i] = gAbs_u8.input0[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gAbs_u16.output0[i] = gAbs_u16.input0[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gAbs_u32.output0[i] = gAbs_u32.input0[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gAbs_u64.output0[i] = gAbs_u64.input0[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gAbs_bool.output0[i] = gAbs_bool.input0[i];
    }
}
