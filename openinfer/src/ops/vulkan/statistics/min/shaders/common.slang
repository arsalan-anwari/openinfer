#ifndef OPENINFER_MIN_COMMON_SLANG
#define OPENINFER_MIN_COMMON_SLANG 1

struct MinPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_MIN_COMMON_SLANG
