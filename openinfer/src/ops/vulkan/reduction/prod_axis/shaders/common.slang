#ifndef OPENINFER_PROD_AXIS_COMMON_SLANG
#define OPENINFER_PROD_AXIS_COMMON_SLANG 1

struct ProdAxisPush {
    uint input_len;
    uint output_len;
    uint input_rank;
    uint output_rank;
    uint axes_mask;
    uint keepdims;
    uint _pad0;
    uint _pad1;
};

#endif // OPENINFER_PROD_AXIS_COMMON_SLANG
