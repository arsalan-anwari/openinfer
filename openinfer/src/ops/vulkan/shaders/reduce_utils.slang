#ifndef OPENINFER_REDUCE_UTILS_SLANG
#define OPENINFER_REDUCE_UTILS_SLANG 1

void linear_to_coords(uint linear, TensorDesc desc, out uint coords[OPENINFER_VK_MAX_DIMS]) {
    uint idx = linear;
    for (int i = int(desc.rank) - 1; i >= 0; --i) {
        uint dim = desc.shape[i];
        uint coord = (dim == 0u) ? 0u : (idx % dim);
        idx = (dim == 0u) ? 0u : (idx / dim);
        coords[i] = coord;
    }
}

bool coords_match(
    uint in_rank,
    uint out_rank,
    uint axes_mask,
    uint keepdims,
    uint in_coords[OPENINFER_VK_MAX_DIMS],
    uint out_coords[OPENINFER_VK_MAX_DIMS]
) {
    uint out_i = 0u;
    for (uint i = 0u; i < in_rank; ++i) {
        bool reduced = ((axes_mask >> i) & 1u) != 0u;
        if (!reduced) {
            uint out_coord = keepdims != 0u ? out_coords[i] : out_coords[out_i++];
            if (in_coords[i] != out_coord) {
                return false;
            }
        }
    }
    return true;
}

#endif // OPENINFER_REDUCE_UTILS_SLANG
