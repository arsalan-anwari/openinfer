#ifndef OPENINFER_COMMON_SLANG
#define OPENINFER_COMMON_SLANG 1

#include "float_utils.slang"
#include "generated_config.slang"

struct TensorDesc {
    uint rank;
    uint dtype;
    uint elem_bits;
    uint byte_offset;
    uint shape[OPENINFER_VK_MAX_DIMS];
    uint strides[OPENINFER_VK_MAX_DIMS];
};

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

#ifndef HAS_F64
#define HAS_F64 0
#endif

uint load_u8(ByteAddressBuffer buf, uint byte_offset) {
    uint word = buf.Load(byte_offset & ~3u);
    uint shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

void store_u8(RWByteAddressBuffer buf, uint byte_offset, uint v) {
    uint addr = byte_offset & ~3u;
    uint word = buf.Load(addr);
    uint shift = (byte_offset & 3u) * 8u;
    uint mask = 0xFFu << shift;
    word = (word & ~mask) | ((v & 0xFFu) << shift);
    buf.Store(addr, word);
}

uint load_u16(ByteAddressBuffer buf, uint byte_offset) {
    uint word = buf.Load(byte_offset & ~3u);
    uint shift = (byte_offset & 2u) * 8u;
    return (word >> shift) & 0xFFFFu;
}

void store_u16(RWByteAddressBuffer buf, uint byte_offset, uint v) {
    uint addr = byte_offset & ~3u;
    uint word = buf.Load(addr);
    uint shift = (byte_offset & 2u) * 8u;
    uint mask = 0xFFFFu << shift;
    word = (word & ~mask) | ((v & 0xFFFFu) << shift);
    buf.Store(addr, word);
}

uint load_bool(ByteAddressBuffer buf, uint byte_offset) {
    return load_u8(buf, byte_offset) & 1u;
}

void store_bool(RWByteAddressBuffer buf, uint byte_offset, uint v) {
    store_u8(buf, byte_offset, v & 1u);
}

uint load_bitset(ByteAddressBuffer buf, uint byte_offset) {
    return load_u8(buf, byte_offset);
}

void store_bitset(RWByteAddressBuffer buf, uint byte_offset, uint v) {
    store_u8(buf, byte_offset, v & 0xFFu);
}

uint elem_byte_stride(TensorDesc desc) {
    return (desc.elem_bits + 7u) / 8u;
}

uint linear_offset(TensorDesc desc, uint linear_index) {
    uint offset = desc.byte_offset;
    uint idx = linear_index;
    uint stride = elem_byte_stride(desc);
    for (int i = int(desc.rank) - 1; i >= 0; --i) {
        uint dim = desc.shape[i];
        uint coord = (dim == 0u) ? 0u : (idx % dim);
        idx = (dim == 0u) ? 0u : (idx / dim);
        offset += coord * desc.strides[i] * stride;
    }
    return offset;
}

float load_f32(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return asfloat(buf.Load(linear_offset(desc, idx)));
}

void store_f32(TensorDesc desc, uint idx, float v, RWByteAddressBuffer buf) {
    buf.Store(linear_offset(desc, idx), asuint(v));
}

float load_f16(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return f16_to_f32(load_u16(buf, linear_offset(desc, idx)));
}

void store_f16(TensorDesc desc, uint idx, float v, RWByteAddressBuffer buf) {
    store_u16(buf, linear_offset(desc, idx), f32_to_f16(v));
}

float load_bf16(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return bf16_to_f32(load_u16(buf, linear_offset(desc, idx)));
}

void store_bf16(TensorDesc desc, uint idx, float v, RWByteAddressBuffer buf) {
    store_u16(buf, linear_offset(desc, idx), f32_to_bf16(v));
}

float load_f8(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return f8_to_f32(load_u8(buf, linear_offset(desc, idx)));
}

void store_f8(TensorDesc desc, uint idx, float v, RWByteAddressBuffer buf) {
    store_u8(buf, linear_offset(desc, idx), f32_to_f8(v));
}

int load_i32(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return asint(buf.Load(linear_offset(desc, idx)));
}

void store_i32(TensorDesc desc, uint idx, int v, RWByteAddressBuffer buf) {
    buf.Store(linear_offset(desc, idx), asuint(v));
}

uint load_u32(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return buf.Load(linear_offset(desc, idx));
}

void store_u32(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    buf.Store(linear_offset(desc, idx), v);
}

int load_i16(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return int(load_u16(buf, linear_offset(desc, idx)));
}

void store_i16(TensorDesc desc, uint idx, int v, RWByteAddressBuffer buf) {
    store_u16(buf, linear_offset(desc, idx), uint(v));
}

uint load_u16_val(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return load_u16(buf, linear_offset(desc, idx));
}

void store_u16_val(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    store_u16(buf, linear_offset(desc, idx), v);
}

int load_i8(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return int(load_u8(buf, linear_offset(desc, idx)));
}

void store_i8(TensorDesc desc, uint idx, int v, RWByteAddressBuffer buf) {
    store_u8(buf, linear_offset(desc, idx), uint(v));
}

uint load_u8_val(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return load_u8(buf, linear_offset(desc, idx));
}

void store_u8_val(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    store_u8(buf, linear_offset(desc, idx), v);
}

uint load_bool_at(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return load_bool(buf, linear_offset(desc, idx));
}

void store_bool_at(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    store_bool(buf, linear_offset(desc, idx), v);
}

uint load_bitset_at(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    return load_bitset(buf, linear_offset(desc, idx));
}

void store_bitset_at(TensorDesc desc, uint idx, uint v, RWByteAddressBuffer buf) {
    store_bitset(buf, linear_offset(desc, idx), v);
}

#if HAS_I64
int64_t load_i64(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    uint offset = linear_offset(desc, idx);
    uint lo = buf.Load(offset);
    uint hi = buf.Load(offset + 4u);
    uint64_t value = (uint64_t(hi) << 32u) | uint64_t(lo);
    return int64_t(value);
}

void store_i64(TensorDesc desc, uint idx, int64_t v, RWByteAddressBuffer buf) {
    uint offset = linear_offset(desc, idx);
    uint64_t value = uint64_t(v);
    buf.Store(offset, uint(value & 0xFFFFFFFFu));
    buf.Store(offset + 4u, uint((value >> 32u) & 0xFFFFFFFFu));
}
#endif

#if HAS_U64
uint64_t load_u64(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    uint offset = linear_offset(desc, idx);
    uint lo = buf.Load(offset);
    uint hi = buf.Load(offset + 4u);
    return (uint64_t(hi) << 32u) | uint64_t(lo);
}

void store_u64(TensorDesc desc, uint idx, uint64_t v, RWByteAddressBuffer buf) {
    uint offset = linear_offset(desc, idx);
    buf.Store(offset, uint(v & 0xFFFFFFFFu));
    buf.Store(offset + 4u, uint((v >> 32u) & 0xFFFFFFFFu));
}
#endif

#if HAS_F64
double load_f64(TensorDesc desc, uint idx, ByteAddressBuffer buf) {
    uint offset = linear_offset(desc, idx);
    uint lo = buf.Load(offset);
    uint hi = buf.Load(offset + 4u);
    uint2 bits = uint2(lo, hi);
    return asdouble(bits);
}

void store_f64(TensorDesc desc, uint idx, double v, RWByteAddressBuffer buf) {
    uint offset = linear_offset(desc, idx);
    uint2 bits = asuint(v);
    buf.Store(offset, bits.x);
    buf.Store(offset + 4u, bits.y);
}
#endif

#endif // OPENINFER_COMMON_SLANG
