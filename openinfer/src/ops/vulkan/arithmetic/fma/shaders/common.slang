#ifndef OPENINFER_FMA_COMMON_SLANG
#define OPENINFER_FMA_COMMON_SLANG 1

struct FmaPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_FMA_COMMON_SLANG
