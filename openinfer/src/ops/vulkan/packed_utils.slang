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
    uint shift = (byte_index & 3u) * 8u;
    uint mask = 0xFFu << shift;
    uint expected;
    uint desired;
    uint original;
    do {
        expected = buf.Load(addr);
        desired = (expected & ~mask) | ((value & 0xFFu) << shift);
        buf.InterlockedCompareExchange(addr, expected, desired, original);
    } while (original != expected);
}

void store_u8_masked(RWByteAddressBuffer buf, uint byte_index, uint mask, uint value) {
    uint addr = byte_index & ~3u;
    uint shift = (byte_index & 3u) * 8u;
    uint byte_mask = (mask & 0xFFu) << shift;
    uint expected;
    uint desired;
    uint original;
    do {
        expected = buf.Load(addr);
        desired = (expected & ~byte_mask) | ((value & mask) << shift);
        buf.InterlockedCompareExchange(addr, expected, desired, original);
    } while (original != expected);
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
    uint sign_bit = 1u << (bits - 1u);
    int signed_val = (int)((raw ^ sign_bit) - sign_bit);
    return signed_val;
}

void packed_write(RWByteAddressBuffer buf, uint index, uint bits, uint raw) {
    uint bit_index = index * bits;
    uint byte_index = bit_index >> 3;
    uint shift = bit_index & 7u;
    uint mask = packed_mask(bits);
    uint low_mask = mask << shift;
    uint low_value = (raw << shift) & low_mask;
    store_u8_masked(buf, byte_index, low_mask, low_value);
    if (shift + bits > 8u) {
        uint high_shift = 8u - shift;
        uint high_mask = mask >> high_shift;
        uint high_value = (raw >> high_shift) & high_mask;
        store_u8_masked(buf, byte_index + 1u, high_mask, high_value);
    }
}
