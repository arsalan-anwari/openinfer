use std::arch::x86_64::*;

use crate::ops::cpu::packed::{packed_mask, packed_per_byte, sign_extend};
use crate::tensor::{I2, I4, U2, U4};

#[allow(dead_code)]
fn unpack_u4_bytes(data: &[U4], logical_len: usize) -> Vec<u8> {
    let per = packed_per_byte(4);
    let max_len = logical_len.min(data.len().saturating_mul(per));
    let mut out = vec![0u8; max_len];
    let max_bytes = (max_len + per - 1) / per;
    unsafe {
        let mask = _mm_set1_epi8(0x0F_u8 as i8);
        let mut byte_index = 0usize;
        while byte_index + 16 <= max_bytes {
            let bytes = _mm_loadu_si128(data.as_ptr().add(byte_index) as *const __m128i);
            let lo = _mm_and_si128(bytes, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            let out_ptr = out.as_mut_ptr().add(byte_index * 2);
            let inter_lo = _mm_unpacklo_epi8(lo, hi);
            let inter_hi = _mm_unpackhi_epi8(lo, hi);
            _mm_storeu_si128(out_ptr as *mut __m128i, inter_lo);
            _mm_storeu_si128(out_ptr.add(16) as *mut __m128i, inter_hi);
            byte_index += 16;
        }
        while byte_index < max_bytes {
            let raw = data[byte_index].bits;
            let base = byte_index * 2;
            if base < max_len {
                out[base] = raw & 0x0F;
            }
            if base + 1 < max_len {
                out[base + 1] = (raw >> 4) & 0x0F;
            }
            byte_index += 1;
        }
    }
    out
}

#[allow(dead_code)]
fn unpack_u2_bytes(data: &[U2], logical_len: usize) -> Vec<u8> {
    let per = packed_per_byte(2);
    let max_len = logical_len.min(data.len().saturating_mul(per));
    let mut out = vec![0u8; max_len];
    let max_bytes = (max_len + per - 1) / per;
    unsafe {
        let mask = _mm_set1_epi8(0x03_u8 as i8);
        let mut byte_index = 0usize;
        while byte_index + 16 <= max_bytes {
            let bytes = _mm_loadu_si128(data.as_ptr().add(byte_index) as *const __m128i);
            let v0 = _mm_and_si128(bytes, mask);
            let v1 = _mm_and_si128(_mm_srli_epi16(bytes, 2), mask);
            let v2 = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            let v3 = _mm_and_si128(_mm_srli_epi16(bytes, 6), mask);
            let t0 = _mm_unpacklo_epi8(v0, v1);
            let t1 = _mm_unpackhi_epi8(v0, v1);
            let t2 = _mm_unpacklo_epi8(v2, v3);
            let t3 = _mm_unpackhi_epi8(v2, v3);
            let o0 = _mm_unpacklo_epi16(t0, t2);
            let o1 = _mm_unpackhi_epi16(t0, t2);
            let o2 = _mm_unpacklo_epi16(t1, t3);
            let o3 = _mm_unpackhi_epi16(t1, t3);
            let out_ptr = out.as_mut_ptr().add(byte_index * 4);
            _mm_storeu_si128(out_ptr as *mut __m128i, o0);
            _mm_storeu_si128(out_ptr.add(16) as *mut __m128i, o1);
            _mm_storeu_si128(out_ptr.add(32) as *mut __m128i, o2);
            _mm_storeu_si128(out_ptr.add(48) as *mut __m128i, o3);
            byte_index += 16;
        }
        while byte_index < max_bytes {
            let raw = data[byte_index].bits;
            let base = byte_index * 4;
            if base < max_len {
                out[base] = raw & 0x03;
            }
            if base + 1 < max_len {
                out[base + 1] = (raw >> 2) & 0x03;
            }
            if base + 2 < max_len {
                out[base + 2] = (raw >> 4) & 0x03;
            }
            if base + 3 < max_len {
                out[base + 3] = (raw >> 6) & 0x03;
            }
            byte_index += 1;
        }
    }
    out
}

#[allow(dead_code)]
fn pack_u4_bytes(values: &[u8], logical_len: usize) -> Vec<U4> {
    let per = packed_per_byte(4);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![U4 { bits: 0 }; storage_len];
    let mask = packed_mask(4);
    for idx in 0..logical_len {
        let raw = values[idx] & mask;
        let byte_idx = idx / per;
        let shift = (idx % per) as u8 * 4;
        let current = out[byte_idx].bits;
        out[byte_idx].bits = (current & !(mask << shift)) | ((raw & mask) << shift);
    }
    out
}

#[allow(dead_code)]
fn pack_u2_bytes(values: &[u8], logical_len: usize) -> Vec<U2> {
    let per = packed_per_byte(2);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![U2 { bits: 0 }; storage_len];
    let mask = packed_mask(2);
    for idx in 0..logical_len {
        let raw = values[idx] & mask;
        let byte_idx = idx / per;
        let shift = (idx % per) as u8 * 2;
        let current = out[byte_idx].bits;
        out[byte_idx].bits = (current & !(mask << shift)) | ((raw & mask) << shift);
    }
    out
}

#[allow(dead_code)]
pub(crate) fn unpack_u4_to_u8(data: &[U4], logical_len: usize) -> Vec<u8> {
    unpack_u4_bytes(data, logical_len)
}

#[allow(dead_code)]
pub(crate) fn unpack_u2_to_u8(data: &[U2], logical_len: usize) -> Vec<u8> {
    unpack_u2_bytes(data, logical_len)
}

#[allow(dead_code)]
pub(crate) fn unpack_i4_to_i8(data: &[I4], logical_len: usize) -> Vec<i8> {
    let packed: Vec<U4> = data.iter().map(|v| U4 { bits: v.bits }).collect();
    let raw = unpack_u4_bytes(&packed, logical_len);
    raw.into_iter()
        .map(|v| sign_extend(v, 4))
        .collect()
}

#[allow(dead_code)]
pub(crate) fn unpack_i2_to_i8(data: &[I2], logical_len: usize) -> Vec<i8> {
    let packed: Vec<U2> = data.iter().map(|v| U2 { bits: v.bits }).collect();
    let raw = unpack_u2_bytes(&packed, logical_len);
    raw.into_iter()
        .map(|v| sign_extend(v, 2))
        .collect()
}

#[allow(dead_code)]
pub(crate) fn pack_u4_from_u8(values: &[u8], logical_len: usize) -> Vec<U4> {
    pack_u4_bytes(values, logical_len)
}

#[allow(dead_code)]
pub(crate) fn pack_u2_from_u8(values: &[u8], logical_len: usize) -> Vec<U2> {
    pack_u2_bytes(values, logical_len)
}

#[allow(dead_code)]
pub(crate) fn pack_i4_from_i8(values: &[i8], logical_len: usize) -> Vec<I4> {
    let mask = packed_mask(4);
    let mut raw = Vec::with_capacity(logical_len);
    for value in values.iter().take(logical_len) {
        raw.push((*value as u8) & mask);
    }
    let packed = pack_u4_bytes(&raw, logical_len);
    packed.into_iter().map(|v| I4 { bits: v.bits }).collect()
}

#[allow(dead_code)]
pub(crate) fn pack_i2_from_i8(values: &[i8], logical_len: usize) -> Vec<I2> {
    let mask = packed_mask(2);
    let mut raw = Vec::with_capacity(logical_len);
    for value in values.iter().take(logical_len) {
        raw.push((*value as u8) & mask);
    }
    let packed = pack_u2_bytes(&raw, logical_len);
    packed.into_iter().map(|v| I2 { bits: v.bits }).collect()
}

pub(crate) fn get_u4_value(data: &[U4], idx: usize) -> u8 {
    let per = packed_per_byte(4);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 4;
    (data[byte_idx].bits >> shift) & packed_mask(4)
}

pub(crate) fn get_u2_value(data: &[U2], idx: usize) -> u8 {
    let per = packed_per_byte(2);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 2;
    (data[byte_idx].bits >> shift) & packed_mask(2)
}

pub(crate) fn get_i4_value(data: &[I4], idx: usize) -> i8 {
    let per = packed_per_byte(4);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 4;
    let raw = (data[byte_idx].bits >> shift) & packed_mask(4);
    sign_extend(raw, 4)
}

pub(crate) fn get_i2_value(data: &[I2], idx: usize) -> i8 {
    let per = packed_per_byte(2);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 2;
    let raw = (data[byte_idx].bits >> shift) & packed_mask(2);
    sign_extend(raw, 2)
}

pub(crate) fn set_u4_value(out: &mut [U4], idx: usize, value: u8) {
    let per = packed_per_byte(4);
    let mask = packed_mask(4);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 4;
    let current = out[byte_idx].bits;
    out[byte_idx].bits = (current & !(mask << shift)) | ((value & mask) << shift);
}

pub(crate) fn set_u2_value(out: &mut [U2], idx: usize, value: u8) {
    let per = packed_per_byte(2);
    let mask = packed_mask(2);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 2;
    let current = out[byte_idx].bits;
    out[byte_idx].bits = (current & !(mask << shift)) | ((value & mask) << shift);
}

pub(crate) fn set_i4_value(out: &mut [I4], idx: usize, value: i8) {
    let per = packed_per_byte(4);
    let mask = packed_mask(4);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 4;
    let current = out[byte_idx].bits;
    let raw = (value as u8) & mask;
    out[byte_idx].bits = (current & !(mask << shift)) | (raw << shift);
}

pub(crate) fn set_i2_value(out: &mut [I2], idx: usize, value: i8) {
    let per = packed_per_byte(2);
    let mask = packed_mask(2);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * 2;
    let current = out[byte_idx].bits;
    let raw = (value as u8) & mask;
    out[byte_idx].bits = (current & !(mask << shift)) | (raw << shift);
}
