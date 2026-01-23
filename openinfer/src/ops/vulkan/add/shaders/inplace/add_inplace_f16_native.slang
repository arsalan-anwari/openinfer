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

struct AddBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<half, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<half, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<half, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<AddBuffers> gAdd;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_inplace_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a_idx = broadcastOffset(i, gAdd.meta0);
    uint b_idx = broadcastOffset(i, gAdd.meta1);
    half a = gAdd.input0[a_idx];
    half b = gAdd.input1[b_idx];
    gAdd.output0[i] = a + b;
}
}
