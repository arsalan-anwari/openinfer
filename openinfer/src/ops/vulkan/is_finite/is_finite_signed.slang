// Is-finite kernel (signed integers).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct IsFiniteScalarBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<IsFiniteScalarBuffers<int>> gIsFinite_i8;
ParameterBlock<IsFiniteScalarBuffers<int>> gIsFinite_i16;
ParameterBlock<IsFiniteScalarBuffers<int>> gIsFinite_i32;
ParameterBlock<IsFiniteScalarBuffers<int64_t>> gIsFinite_i64;

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_i8.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_i16.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_i32.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_i64.output0[0] = 1u;
}
