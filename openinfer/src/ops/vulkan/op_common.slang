struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

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
