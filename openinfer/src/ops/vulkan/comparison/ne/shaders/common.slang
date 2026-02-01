#ifndef OPENINFER_NE_COMMON_SLANG
#define OPENINFER_NE_COMMON_SLANG 1

struct NePush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_NE_COMMON_SLANG
