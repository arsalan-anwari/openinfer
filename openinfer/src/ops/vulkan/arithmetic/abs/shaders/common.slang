#ifndef OPENINFER_ABS_COMMON_SLANG
#define OPENINFER_ABS_COMMON_SLANG 1

struct AbsPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_ABS_COMMON_SLANG
