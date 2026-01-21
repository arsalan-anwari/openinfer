use crate::tensor::{I1, I2, I4, U1, U2, U4};

pub(crate) trait PackedByte: Copy {
    fn get(self) -> u8;
    fn set(&mut self, value: u8);
}

impl PackedByte for I4 {
    fn get(self) -> u8 {
        self.bits
    }

    fn set(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedByte for I2 {
    fn get(self) -> u8 {
        self.bits
    }

    fn set(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedByte for I1 {
    fn get(self) -> u8 {
        self.bits
    }

    fn set(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedByte for U4 {
    fn get(self) -> u8 {
        self.bits
    }

    fn set(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedByte for U2 {
    fn get(self) -> u8 {
        self.bits
    }

    fn set(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedByte for U1 {
    fn get(self) -> u8 {
        self.bits
    }

    fn set(&mut self, value: u8) {
        self.bits = value;
    }
}

pub(crate) fn packed_mask(bits: u8) -> u8 {
    ((1u16 << bits) - 1) as u8
}

pub(crate) fn packed_per_byte(bits: u8) -> usize {
    (8 / bits) as usize
}

pub(crate) fn packed_read<T: PackedByte>(data: &[T], idx: usize, bits: u8) -> u8 {
    let per = packed_per_byte(bits);
    let byte = data[idx / per].get();
    let shift = (idx % per) as u8 * bits;
    (byte >> shift) & packed_mask(bits)
}

pub(crate) fn packed_write<T: PackedByte>(data: &mut [T], idx: usize, bits: u8, raw: u8) {
    let per = packed_per_byte(bits);
    let byte_idx = idx / per;
    let shift = (idx % per) as u8 * bits;
    let mask = packed_mask(bits) << shift;
    let current = data[byte_idx].get();
    let next = (current & !mask) | ((raw << shift) & mask);
    let mut slot = data[byte_idx];
    slot.set(next);
    data[byte_idx] = slot;
}

pub(crate) fn sign_extend(raw: u8, bits: u8) -> i8 {
    let shift = 8 - bits;
    ((raw << shift) as i8) >> shift
}

pub(crate) fn packed_binary_signed<T: PackedByte>(
    bits: u8,
    a: &[T],
    b: &[T],
    logical_len: usize,
    zero: T,
    op: impl Fn(i8, i8) -> i8,
) -> Vec<T> {
    let per = packed_per_byte(bits);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![zero; storage_len];
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(a, idx, bits), bits);
        let y = sign_extend(packed_read(b, idx, bits), bits);
        let raw = op(x, y) as u8;
        packed_write(&mut out, idx, bits, raw);
    }
    out
}

pub(crate) fn packed_binary_unsigned<T: PackedByte>(
    bits: u8,
    a: &[T],
    b: &[T],
    logical_len: usize,
    zero: T,
    op: impl Fn(u8, u8) -> u8,
) -> Vec<T> {
    let per = packed_per_byte(bits);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![zero; storage_len];
    for idx in 0..logical_len {
        let x = packed_read(a, idx, bits);
        let y = packed_read(b, idx, bits);
        let raw = op(x, y);
        packed_write(&mut out, idx, bits, raw);
    }
    out
}

pub(crate) fn packed_unary_signed<T: PackedByte>(
    bits: u8,
    a: &[T],
    logical_len: usize,
    zero: T,
    op: impl Fn(i8) -> i8,
) -> Vec<T> {
    let per = packed_per_byte(bits);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![zero; storage_len];
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(a, idx, bits), bits);
        let raw = op(x) as u8;
        packed_write(&mut out, idx, bits, raw);
    }
    out
}

pub(crate) fn packed_binary_accumulate_signed<T: PackedByte, O>(
    bits: u8,
    a: &[T],
    b: &[T],
    logical_len: usize,
    op: impl Fn(i8, i8) -> O,
) -> Vec<O> {
    let mut out = Vec::with_capacity(logical_len);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(a, idx, bits), bits);
        let y = sign_extend(packed_read(b, idx, bits), bits);
        out.push(op(x, y));
    }
    out
}

pub(crate) fn packed_binary_accumulate_unsigned<T: PackedByte, O>(
    bits: u8,
    a: &[T],
    b: &[T],
    logical_len: usize,
    op: impl Fn(u8, u8) -> O,
) -> Vec<O> {
    let mut out = Vec::with_capacity(logical_len);
    for idx in 0..logical_len {
        let x = packed_read(a, idx, bits);
        let y = packed_read(b, idx, bits);
        out.push(op(x, y));
    }
    out
}

pub(crate) fn packed_unary_accumulate_signed<T: PackedByte, O>(
    bits: u8,
    a: &[T],
    logical_len: usize,
    op: impl Fn(i8) -> O,
) -> Vec<O> {
    let mut out = Vec::with_capacity(logical_len);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(a, idx, bits), bits);
        out.push(op(x));
    }
    out
}

pub(crate) fn packed_fill_signed<T: PackedByte>(
    bits: u8,
    logical_len: usize,
    value: i32,
    zero: T,
) -> Vec<T> {
    let per = packed_per_byte(bits);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![zero; storage_len];
    let raw = value as u8;
    for idx in 0..logical_len {
        packed_write(&mut out, idx, bits, raw);
    }
    out
}

pub(crate) fn packed_fill_unsigned<T: PackedByte>(
    bits: u8,
    logical_len: usize,
    value: u32,
    zero: T,
) -> Vec<T> {
    let per = packed_per_byte(bits);
    let storage_len = (logical_len + per - 1) / per;
    let mut out = vec![zero; storage_len];
    let raw = value as u8;
    for idx in 0..logical_len {
        packed_write(&mut out, idx, bits, raw);
    }
    out
}
