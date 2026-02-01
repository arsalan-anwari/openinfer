#ifndef OPENINFER_POPCOUNT_COMMON_SLANG
#define OPENINFER_POPCOUNT_COMMON_SLANG 1

struct PopcountPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_POPCOUNT_COMMON_SLANG
