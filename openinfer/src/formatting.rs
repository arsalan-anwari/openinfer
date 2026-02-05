use crate::tensor::{BF16, Bitset, F16, F8, I1, I2, I4, T1, T2, U1, U2, U4};
/// Format a value for compact human-readable output.
pub trait FormatValue {
    fn format_value(&self) -> String;
}

impl FormatValue for f32 {
    fn format_value(&self) -> String {
        format!("{:.2}", self)
    }
}

impl FormatValue for f64 {
    fn format_value(&self) -> String {
        format!("{:.2}", self)
    }
}

macro_rules! impl_format_display {
    ($($ty:ty),+ $(,)?) => {
        $(impl FormatValue for $ty {
            fn format_value(&self) -> String {
                self.to_string()
            }
        })+
    };
}

impl_format_display!(i8, i16, i32, i64, u8, u16, u32, u64, bool);

impl FormatValue for Bitset {
    fn format_value(&self) -> String {
        format!("{:?}", self)
    }
}

impl FormatValue for F16 {
    fn format_value(&self) -> String {
        format!("{:?}", self)
    }
}

impl FormatValue for BF16 {
    fn format_value(&self) -> String {
        format!("{:.2}", self.to_f32())
    }
}

impl FormatValue for F8 {
    fn format_value(&self) -> String {
        format!("{:.2}", self.to_f32())
    }
}

impl FormatValue for I4 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for I2 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for I1 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for U4 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for U2 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for U1 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for T2 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

impl FormatValue for T1 {
    fn format_value(&self) -> String {
        format!("0x{:02x}", self.bits)
    }
}

/// Format a slice with head/tail truncation.
pub fn format_truncated<T: FormatValue>(data: &[T]) -> String {
    let len = data.len();
    if len == 0 {
        return "{}".to_string();
    }
    if len <= 4 {
        let joined = data
            .iter()
            .map(FormatValue::format_value)
            .collect::<Vec<_>>()
            .join(", ");
        return format!("{{{}}}", joined);
    }
    let head = &data[..2];
    let tail = &data[len - 2..];
    format!(
        "{{{}, {} ... {}, {}}}",
        head[0].format_value(),
        head[1].format_value(),
        tail[0].format_value(),
        tail[1].format_value()
    )
}
