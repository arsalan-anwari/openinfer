#ifndef OPENINFER_MAX_COMMON_SLANG
#define OPENINFER_MAX_COMMON_SLANG 1

struct MaxPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_MAX_COMMON_SLANG
