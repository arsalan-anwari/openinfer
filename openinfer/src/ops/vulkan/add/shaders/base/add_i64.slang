struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

uint broadcastOffset(uint index, StructuredBuffer<uint, Std430DataLayout> meta) {
    uint rank = meta[0];
    uint baseOut = 2;
    uint baseInShape = 2 + rank;
    uint baseInStride = 2 + rank * 2;
    uint rem = index;
    uint offset = 0;
    if (rank > 0) {
        for (int r = int(rank) - 1; r >= 0; --r) {
            uint ur = (uint)r;
            uint dim = meta[baseOut + ur];
            uint coord = 0;
            if (dim > 0) {
                coord = rem % dim;
                rem = rem / dim;
            }
            uint inDim = meta[baseInShape + ur];
            uint stride = meta[baseInStride + ur];
            uint inCoord = (inDim == 1) ? 0 : coord;
            offset += inCoord * stride;
        }
    }
    return offset;
}

struct AddBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<AddBuffers<int64_t>> gAdd_int64_t;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a_idx = broadcastOffset(i, gAdd_int64_t.meta0);
    uint b_idx = broadcastOffset(i, gAdd_int64_t.meta1);
    gAdd_int64_t.output0[i] = gAdd_int64_t.input0[a_idx] + gAdd_int64_t.input1[b_idx];
}
}
