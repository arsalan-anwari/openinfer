// Generic broadcast kernel (single source, multiple entry points).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct BroadcastBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> meta;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<BroadcastBuffers<float>> gBroadcast_f32;
ParameterBlock<BroadcastBuffers<int>> gBroadcast_i8;
ParameterBlock<BroadcastBuffers<int>> gBroadcast_i16;
ParameterBlock<BroadcastBuffers<int>> gBroadcast_i32;
ParameterBlock<BroadcastBuffers<uint>> gBroadcast_u8;
ParameterBlock<BroadcastBuffers<uint>> gBroadcast_u16;
ParameterBlock<BroadcastBuffers<uint>> gBroadcast_u32;
ParameterBlock<BroadcastBuffers<uint>> gBroadcast_bool;
ParameterBlock<BroadcastBuffers<int64_t>> gBroadcast_i64;
ParameterBlock<BroadcastBuffers<uint64_t>> gBroadcast_u64;

__generic<typename T, let BLOCK_SIZE : int>
void broadcastImpl(uint3 tid, BroadcastBuffers<T> bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i >= pc.len) {
        return;
    }
    uint rank = bufs.meta[0];
    uint baseOut = 2;
    uint baseInShape = 2 + rank;
    uint baseInStride = 2 + rank * 2;
    uint rem = i;
    uint offset = 0;
    if (rank > 0) {
        for (int r = int(rank) - 1; r >= 0; --r) {
            uint ur = (uint)r;
            uint dim = bufs.meta[baseOut + ur];
            uint coord = 0;
            if (dim > 0) {
                coord = rem % dim;
                rem = rem / dim;
            }
            uint inDim = bufs.meta[baseInShape + ur];
            uint stride = bufs.meta[baseInStride + ur];
            uint inCoord = (inDim == 1) ? 0 : coord;
            offset += inCoord * stride;
        }
    }
    bufs.output0[i] = bufs.input0[offset];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<float, 256>(tid, gBroadcast_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<int, 256>(tid, gBroadcast_i8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<int, 256>(tid, gBroadcast_i16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<int, 256>(tid, gBroadcast_i32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<uint, 256>(tid, gBroadcast_u8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<uint, 256>(tid, gBroadcast_u16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<uint, 256>(tid, gBroadcast_u32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<uint, 256>(tid, gBroadcast_bool, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<int64_t, 256>(tid, gBroadcast_i64, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void broadcast_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    broadcastImpl<uint64_t, 256>(tid, gBroadcast_u64, pc);
}
