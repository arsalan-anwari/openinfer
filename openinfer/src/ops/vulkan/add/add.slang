// Generic add kernel (single source, multiple entry points).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<AddBuffers<float>> gAdd_f32;
ParameterBlock<AddBuffers<int>> gAdd_i8;
ParameterBlock<AddBuffers<int>> gAdd_i16;
ParameterBlock<AddBuffers<int>> gAdd_i32;
ParameterBlock<AddBuffers<uint>> gAdd_u8;
ParameterBlock<AddBuffers<uint>> gAdd_u16;
ParameterBlock<AddBuffers<uint>> gAdd_u32;
ParameterBlock<AddBuffers<uint>> gAdd_bool;
ParameterBlock<AddBuffers<int64_t>> gAdd_i64;
ParameterBlock<AddBuffers<uint64_t>> gAdd_u64;

__generic<typename T, let BLOCK_SIZE : int>
void addImpl(uint3 tid, AddBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint i = tid.x;
    if (i < pc.len) {
        bufs.output0[i] = bufs.input0[i] + bufs.input1[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<float, 256>(tid, gAdd_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<int, 256>(tid, gAdd_i8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<int, 256>(tid, gAdd_i16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<int, 256>(tid, gAdd_i32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_u8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_u16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_u32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_bool, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<int64_t, 256>(tid, gAdd_i64, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint64_t, 256>(tid, gAdd_u64, pc);
}
