#ifndef OPENINFER_FLOOR_COMMON_SLANG
#define OPENINFER_FLOOR_COMMON_SLANG 1

struct FloorPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_FLOOR_COMMON_SLANG
