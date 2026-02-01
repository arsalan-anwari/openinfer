#ifndef OPENINFER_SHR_COMMON_SLANG
#define OPENINFER_SHR_COMMON_SLANG 1

struct ShrPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    uint bits;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

#endif // OPENINFER_SHR_COMMON_SLANG
