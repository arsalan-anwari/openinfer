// Mul inplace kernel (unsigned integers and bool).
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

ParameterBlock<MulBuffers<uint>> gMul_u8;
ParameterBlock<MulBuffers<uint>> gMul_u16;
ParameterBlock<MulBuffers<uint>> gMul_u32;
ParameterBlock<MulBuffers<uint64_t>> gMul_u64;
ParameterBlock<MulBuffers<uint>> gMul_bool;

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
void mul_inplace_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_u8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_u16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_u32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint64_t, 256>(tid, gMul_u64, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_bool, pc);
}
