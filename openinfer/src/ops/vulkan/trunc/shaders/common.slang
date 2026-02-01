#ifndef OPENINFER_TRUNC_COMMON_SLANG
#define OPENINFER_TRUNC_COMMON_SLANG 1

struct TruncPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_TRUNC_COMMON_SLANG
