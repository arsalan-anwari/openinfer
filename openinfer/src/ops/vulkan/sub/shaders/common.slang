#ifndef OPENINFER_SUB_COMMON_SLANG
#define OPENINFER_SUB_COMMON_SLANG 1

struct SubPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_SUB_COMMON_SLANG
