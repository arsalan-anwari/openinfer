#ifndef OPENINFER_FILTER_COMMON_SLANG
#define OPENINFER_FILTER_COMMON_SLANG 1

struct FilterPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_FILTER_COMMON_SLANG
