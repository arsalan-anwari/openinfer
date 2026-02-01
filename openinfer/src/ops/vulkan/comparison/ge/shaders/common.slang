#ifndef OPENINFER_GE_COMMON_SLANG
#define OPENINFER_GE_COMMON_SLANG 1

struct GePush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_GE_COMMON_SLANG
