#ifndef OPENINFER_CAST_COMMON_SLANG
#define OPENINFER_CAST_COMMON_SLANG 1

struct CastPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_CAST_COMMON_SLANG
