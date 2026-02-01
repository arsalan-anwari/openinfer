#ifndef OPENINFER_MAX_AXIS_COMMON_SLANG
#define OPENINFER_MAX_AXIS_COMMON_SLANG 1

struct MaxAxisPush {
    uint input_len;
    uint output_len;
    uint input_rank;
    uint output_rank;
    uint axes_mask;
    uint keepdims;
    uint _pad0;
    uint _pad1;
};

#endif // OPENINFER_MAX_AXIS_COMMON_SLANG
