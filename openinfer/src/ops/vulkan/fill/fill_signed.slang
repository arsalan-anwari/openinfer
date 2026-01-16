// Fill kernel (signed integers).
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

ParameterBlock<FillBuffers<int>> gFill_i8;
ParameterBlock<FillBuffers<int>> gFill_i16;
ParameterBlock<FillBuffers<int>> gFill_i32;
ParameterBlock<FillBuffers<int64_t>> gFill_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_i8.output0[i] = (int)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_i16.output0[i] = (int)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_i32.output0[i] = (int)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits);
        gFill_i64.output0[i] = (int64_t)value;
    }
}
