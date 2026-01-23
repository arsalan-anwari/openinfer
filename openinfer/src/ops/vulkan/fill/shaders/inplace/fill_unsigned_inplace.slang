struct PushConsts {
    uint len;
    uint value_bits;
    uint pad0;
    uint pad1;
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
void fill_inplace_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_u8.output0[i] = (uint)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_u16.output0[i] = (uint)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_u32.output0[i] = (uint)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_u64.output0[i] = (uint64_t)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_bool.output0[i] = value != 0.0f ? 1u : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_bitset(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        uint v = value == 0.0f ? 0u : 0xFFu;
        gFill_bitset.output0[i] = v;
    }
}
