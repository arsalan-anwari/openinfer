#ifndef OPENINFER_GT_COMMON_SLANG
#define OPENINFER_GT_COMMON_SLANG 1

struct GtPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_GT_COMMON_SLANG
