uint load_u8(ByteAddressBuffer buf, uint byte_index) {
    uint addr = byte_index & ~3u;
    uint word = buf.Load(addr);
    uint shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

uint load_u8_rw(RWByteAddressBuffer buf, uint byte_index) {
    uint addr = byte_index & ~3u;
    uint word = buf.Load(addr);
    uint shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

void store_u8(RWByteAddressBuffer buf, uint byte_index, uint value) {
    uint addr = byte_index & ~3u;
    uint word = buf.Load(addr);
    uint shift = (byte_index & 3u) * 8u;
    uint mask = 0xFFu << shift;
    word = (word & ~mask) | ((value & 0xFFu) << shift);
    buf.Store(addr, word);
}

uint packed_mask(uint bits) {
    return (1u << bits) - 1u;
}

uint packed_read_unsigned(ByteAddressBuffer buf, uint index, uint bits) {
    uint bit_index = index * bits;
    uint byte_index = bit_index >> 3;
    uint shift = bit_index & 7u;
    uint raw = load_u8(buf, byte_index) >> shift;
    if (shift + bits > 8u) {
        raw |= load_u8(buf, byte_index + 1u) << (8u - shift);
    }
    return raw & packed_mask(bits);
}

int packed_read_signed(ByteAddressBuffer buf, uint index, uint bits) {
    uint raw = packed_read_unsigned(buf, index, bits);
    uint shift = 8u - bits;
    return (int)((raw << shift) >> shift);
}

void packed_write(RWByteAddressBuffer buf, uint index, uint bits, uint raw) {
    uint bit_index = index * bits;
    uint byte_index = bit_index >> 3;
    uint shift = bit_index & 7u;
    uint mask = packed_mask(bits);
    uint low_mask = mask << shift;
    uint low = load_u8_rw(buf, byte_index);
    low = (low & ~low_mask) | ((raw << shift) & low_mask);
    store_u8(buf, byte_index, low);
    if (shift + bits > 8u) {
        uint high_shift = 8u - shift;
        uint high_mask = mask >> high_shift;
        uint high = load_u8_rw(buf, byte_index + 1u);
        high = (high & ~high_mask) | ((raw >> high_shift) & high_mask);
        store_u8(buf, byte_index + 1u, high);
    }
}
