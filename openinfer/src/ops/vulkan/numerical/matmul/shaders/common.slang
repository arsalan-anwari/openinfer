#ifndef OPENINFER_MATMUL_COMMON_SLANG
#define OPENINFER_MATMUL_COMMON_SLANG 1

struct MatmulPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_MATMUL_COMMON_SLANG
