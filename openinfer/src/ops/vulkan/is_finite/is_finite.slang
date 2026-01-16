// Is-finite kernel (single-pass, scalar reduction).
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

ParameterBlock<IsFiniteScalarBuffers<float>> gIsFinite_f32;
ParameterBlock<IsFiniteScalarBuffers<int>> gIsFinite_i8;
ParameterBlock<IsFiniteScalarBuffers<int>> gIsFinite_i16;
ParameterBlock<IsFiniteScalarBuffers<int>> gIsFinite_i32;
ParameterBlock<IsFiniteScalarBuffers<uint>> gIsFinite_u8;
ParameterBlock<IsFiniteScalarBuffers<uint>> gIsFinite_u16;
ParameterBlock<IsFiniteScalarBuffers<uint>> gIsFinite_u32;
ParameterBlock<IsFiniteScalarBuffers<uint>> gIsFinite_bool;
ParameterBlock<IsFiniteScalarBuffers<int64_t>> gIsFinite_i64;
ParameterBlock<IsFiniteScalarBuffers<uint64_t>> gIsFinite_u64;

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = gIsFinite_f32.input0[i];
        float abs_x = abs(x);
        if (x != x || abs_x > 3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f32.output0[0] = ok;
}

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
void is_finite_scalar_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_u8.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_u16.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_u32.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_bool.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_i64.output0[0] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    gIsFinite_u64.output0[0] = 1u;
}
