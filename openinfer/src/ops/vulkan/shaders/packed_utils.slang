#ifndef OPENINFER_PACKED_UTILS
#define OPENINFER_PACKED_UTILS 1

uint read_bits(ByteAddressBuffer buf, uint bit_offset, uint bit_width) {
    uint byte_offset = (bit_offset >> 3) & ~3u;
    uint shift = bit_offset - (byte_offset * 8u);
    uint lo = buf.Load(byte_offset);
    uint hi = buf.Load(byte_offset + 4);
    uint combined = (lo >> shift) | (hi << (32u - shift));
    uint mask = (1u << bit_width) - 1u;
    return combined & mask;
}

void write_bits(RWByteAddressBuffer buf, uint bit_offset, uint bit_width, uint value) {
    uint byte_offset = (bit_offset >> 3) & ~3u;
    uint shift = bit_offset - (byte_offset * 8u);
    uint mask = ((1u << bit_width) - 1u) << shift;
    uint word = buf.Load(byte_offset);
    word = (word & ~mask) | ((value << shift) & mask);
    buf.Store(byte_offset, word);
}

uint packed_bit_offset(TensorDesc desc, uint idx) {
    uint offset = desc.byte_offset * 8u;
    uint idx_local = idx;
    for (int i = int(desc.rank) - 1; i >= 0; --i) {
        uint dim = desc.shape[i];
        uint coord = (dim == 0u) ? 0u : (idx_local % dim);
        idx_local = (dim == 0u) ? 0u : (idx_local / dim);
        offset += coord * desc.strides[i] * desc.elem_bits;
    }
    return offset;
}

int sign_extend(uint v, uint bit_width) {
    uint sign_bit = 1u << (bit_width - 1u);
    return (v & sign_bit) != 0u ? int(v | (~((1u << bit_width) - 1u))) : int(v);
}

uint load_packed_u(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return read_bits(buf, packed_bit_offset(desc, idx), desc.elem_bits);
}

int load_packed_i(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    uint v = read_bits(buf, packed_bit_offset(desc, idx), desc.elem_bits);
    return sign_extend(v, desc.elem_bits);
}

void store_packed(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    write_bits(buf, packed_bit_offset(desc, idx), desc.elem_bits, v);
}

#endif // OPENINFER_PACKED_UTILS
