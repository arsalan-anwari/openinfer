use crate::tensor::{I1, I2, I4, U1, U2, U4};

pub trait PackedBits: Copy {
    fn bits(&self) -> u8;
    fn set_bits(&mut self, value: u8);
}

pub fn get_bits<T: PackedBits>(data: &[T], index: usize, width: u8) -> u8 {
    let per_byte = 8 / width;
    let byte_index = index / per_byte as usize;
    let bit_index = (index % per_byte as usize) as u8;
    let shift = bit_index * width;
    let mask = (1u8 << width) - 1;
    let byte = data[byte_index].bits();
    (byte >> shift) & mask
}

pub fn set_bits<T: PackedBits>(data: &mut [T], index: usize, width: u8, value: u8) {
    let per_byte = 8 / width;
    let byte_index = index / per_byte as usize;
    let bit_index = (index % per_byte as usize) as u8;
    let shift = bit_index * width;
    let mask = (1u8 << width) - 1;
    let mut byte = data[byte_index].bits();
    byte &= !(mask << shift);
    byte |= (value & mask) << shift;
    data[byte_index].set_bits(byte);
}

pub fn sign_extend(value: u8, width: u8) -> i8 {
    let shift = 8 - width;
    ((value << shift) as i8) >> shift
}

impl PackedBits for I1 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedBits for I2 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedBits for I4 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedBits for U1 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedBits for U2 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedBits for U4 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}
