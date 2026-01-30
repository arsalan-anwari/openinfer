#ifndef OPENINFER_PACKED_UTILS
#define OPENINFER_PACKED_UTILS 1

#include "common.slang"

// Safe N-bit mask: for n==0 => 0, for n>=32 => 0xFFFFFFFF, else (1<<n)-1
uint mask_n(uint n) {
    return (n >= 32u) ? 0xFFFFFFFFu : ((n == 0u) ? 0u : ((1u << n) - 1u));
}

uint pack_signed_nbit(int x, uint bit_width) {
    return uint(x) & mask_n(bit_width);
}

uint read_bits(ByteAddressBuffer buf, uint bit_offset, uint bit_width) {
    uint byte_offset = (bit_offset >> 3) & ~3u;
    uint shift = bit_offset - (byte_offset * 8u);

    uint lo = buf.Load(byte_offset);

    // Only load hi when needed to avoid potential OOB reads (still recommend padding!)
    uint combined;
    if (shift == 0u) {
        combined = lo;
    } else {
        uint hi = buf.Load(byte_offset + 4u);
        combined = (lo >> shift) | (hi << (32u - shift));
    }

    return combined & mask_n(bit_width);
}

// Sign-extend a bitfield v[bit_width-1:0] to 32-bit signed int
int sign_extend(uint v, uint bit_width) {
    if (bit_width == 0u) return 0;
    if (bit_width >= 32u) return int(v);

    uint m = mask_n(bit_width);
    v &= m;

    uint sign_bit = 1u << (bit_width - 1u);
    if ((v & sign_bit) != 0u) {
        // Fill upper bits with 1s
        return int(v | ~m);
    } else {
        return int(v);
    }
}

void write_bits(RWByteAddressBuffer buf, uint bit_offset, uint bit_width, uint value) {
    if (bit_width == 0u) return;

    value &= mask_n(bit_width);

    uint byte_offset = (bit_offset >> 3) & ~3u;
    uint shift       = bit_offset - (byte_offset * 8u);

    uint lo_bits = min(bit_width, 32u - shift);
    uint hi_bits = bit_width - lo_bits;

    uint lo_mask  = mask_n(lo_bits) << shift;
    uint lo_value = (value & mask_n(lo_bits)) << shift;

    uint current, original;
    do {
        current = buf.Load(byte_offset);
        uint next = (current & ~lo_mask) | lo_value;
        buf.InterlockedCompareExchange(byte_offset, current, next, original);
    } while (original != current);

    if (hi_bits > 0u) {
        uint hi_addr  = byte_offset + 4u;
        uint hi_mask  = mask_n(hi_bits); // starts at bit 0 in next word
        uint hi_value = value >> lo_bits;

        do {
            current = buf.Load(hi_addr);
            uint next = (current & ~hi_mask) | (hi_value & hi_mask);
            buf.InterlockedCompareExchange(hi_addr, current, next, original);
        } while (original != current);
    }
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

uint load_packed_u(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return read_bits(buf, packed_bit_offset(desc, idx), desc.elem_bits);
}

int load_packed_i(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    uint v = read_bits(buf, packed_bit_offset(desc, idx), desc.elem_bits);
    return sign_extend(v, desc.elem_bits);
}

int load_packed_byte_i(TensorDesc desc, uint byte_idx, ByteAddressBuffer buf) {
    uint byte_offset = desc.byte_offset + byte_idx;
    uint v = load_u8(buf, byte_offset);
    return sign_extend(v, desc.elem_bits);
}

void store_packed(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    write_bits(buf, packed_bit_offset(desc, idx), desc.elem_bits, v);
}

#endif // OPENINFER_PACKED_UTILS