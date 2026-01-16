// Mul kernel (signed integers).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<MulBuffers<int>> gMul_i8;
ParameterBlock<MulBuffers<int>> gMul_i16;
ParameterBlock<MulBuffers<int>> gMul_i32;
ParameterBlock<MulBuffers<int64_t>> gMul_i64;

__generic<typename T, let BLOCK_SIZE : int>
void mulImpl(uint3 tid, MulBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint i = tid.x;
    if (i < pc.len) {
        bufs.output0[i] = bufs.input0[i] * bufs.input1[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int, 256>(tid, gMul_i8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int, 256>(tid, gMul_i16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int, 256>(tid, gMul_i32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int64_t, 256>(tid, gMul_i64, pc);
}
