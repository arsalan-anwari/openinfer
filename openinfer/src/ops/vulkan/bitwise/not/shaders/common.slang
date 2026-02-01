#ifndef OPENINFER_NOT_COMMON_SLANG
#define OPENINFER_NOT_COMMON_SLANG 1

struct NotPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_NOT_COMMON_SLANG
