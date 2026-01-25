struct PushConsts {
    uint len;
    uint value_u32;
    uint64_t value_u64;
};

[[vk::push_constant]] PushConsts pc;

struct FillBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<FillBuffers<uint>> gFill_u8;
ParameterBlock<FillBuffers<uint>> gFill_u16;
ParameterBlock<FillBuffers<uint>> gFill_u32;
ParameterBlock<FillBuffers<uint64_t>> gFill_u64;
ParameterBlock<FillBuffers<uint>> gFill_bool;
ParameterBlock<FillBuffers<uint>> gFill_bitset;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_u8.output0[i] = pc.value_u32;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_u16.output0[i] = pc.value_u32;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_u32.output0[i] = pc.value_u32;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_u64.output0[i] = pc.value_u64;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gFill_bool.output0[i] = pc.value_u32 != 0 ? 1u : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_bitset(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint v = pc.value_u32 & 0xFFu;
        gFill_bitset.output0[i] = v;
    }
}
