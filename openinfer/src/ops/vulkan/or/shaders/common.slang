#ifndef OPENINFER_OR_COMMON_SLANG
#define OPENINFER_OR_COMMON_SLANG 1

struct OrPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_OR_COMMON_SLANG
