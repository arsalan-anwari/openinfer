use openinfer::{graph, ModelLoader, Simulator, TensorValue};
use std::path::Path;

mod util;
use util::select_device;

fn format_slice<T: std::fmt::Debug>(data: &[T]) -> String {
    let n = 8.min(data.len());
    format!("{:?}", &data[..n])
}

fn print_tensor_value(name: &str, value: TensorValue) {
    match value {
        TensorValue::I8(t) => openinfer::log!("{name}: i8 {}", format_slice(&t.data)),
        TensorValue::I16(t) => openinfer::log!("{name}: i16 {}", format_slice(&t.data)),
        TensorValue::I32(t) => openinfer::log!("{name}: i32 {}", format_slice(&t.data)),
        TensorValue::I64(t) => openinfer::log!("{name}: i64 {}", format_slice(&t.data)),
        TensorValue::U8(t) => openinfer::log!("{name}: u8 {}", format_slice(&t.data)),
        TensorValue::U16(t) => openinfer::log!("{name}: u16 {}", format_slice(&t.data)),
        TensorValue::U32(t) => openinfer::log!("{name}: u32 {}", format_slice(&t.data)),
        TensorValue::U64(t) => openinfer::log!("{name}: u64 {}", format_slice(&t.data)),
        other => openinfer::log!("{name}: dtype={:?}", other.dtype()),
    }
}

const OUTPUTS: &[&str] = &[
    "add_i8_i16",
    "add_i16_i32",
    "add_i32_i64",
    "add_u8_u16",
    "add_u16_u32",
    "add_u32_u64",
    "add_i4_i8",
    "add_i2_i8",
    "add_i1_i8",
    "add_u4_u8",
    "add_u2_u8",
    "add_u1_u8",
    "mul_i8_i16",
    "mul_i16_i32",
    "mul_i32_i64",
    "mul_u8_u16",
    "mul_u16_u32",
    "mul_u32_u64",
    "mul_i4_i8",
    "mul_i2_i8",
    "mul_i1_i8",
    "mul_u4_u8",
    "mul_u2_u8",
    "mul_u1_u8",
    "abs_i8_i16",
    "abs_i16_i32",
    "abs_i32_i64",
    "abs_i4_i8",
    "abs_i2_i8",
    "abs_i1_i8",
];

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/accumulate_packed_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        constant {
            a_i8: i8[N];
            b_i8: i8[N];
            a_i16: i16[N];
            b_i16: i16[N];
            a_i32: i32[N];
            b_i32: i32[N];
            a_u8: u8[N];
            b_u8: u8[N];
            a_u16: u16[N];
            b_u16: u16[N];
            a_u32: u32[N];
            b_u32: u32[N];
            a_i4: i4[N];
            b_i4: i4[N];
            a_i2: i2[N];
            b_i2: i2[N];
            a_i1: i1[N];
            b_i1: i1[N];
            a_u4: u4[N];
            b_u4: u4[N];
            a_u2: u2[N];
            b_u2: u2[N];
            a_u1: u1[N];
            b_u1: u1[N];
        }

        volatile {
            add_i8_i16: i16[N];
            add_i16_i32: i32[N];
            add_i32_i64: i64[N];
            add_u8_u16: u16[N];
            add_u16_u32: u32[N];
            add_u32_u64: u64[N];
            add_i4_i8: i8[N];
            add_i2_i8: i8[N];
            add_i1_i8: i8[N];
            add_u4_u8: u8[N];
            add_u2_u8: u8[N];
            add_u1_u8: u8[N];
            mul_i8_i16: i16[N];
            mul_i16_i32: i32[N];
            mul_i32_i64: i64[N];
            mul_u8_u16: u16[N];
            mul_u16_u32: u32[N];
            mul_u32_u64: u64[N];
            mul_i4_i8: i8[N];
            mul_i2_i8: i8[N];
            mul_i1_i8: i8[N];
            mul_u4_u8: u8[N];
            mul_u2_u8: u8[N];
            mul_u1_u8: u8[N];
            abs_i8_i16: i16[N];
            abs_i16_i32: i32[N];
            abs_i32_i64: i64[N];
            abs_i4_i8: i8[N];
            abs_i2_i8: i8[N];
            abs_i1_i8: i8[N];
        }

        block entry {
            op add(a_i8, b_i8, acc=i16) >> add_i8_i16;
            op add(a_i16, b_i16, acc=i32) >> add_i16_i32;
            op add(a_i32, b_i32, acc=i64) >> add_i32_i64;
            op add(a_u8, b_u8, acc=u16) >> add_u8_u16;
            op add(a_u16, b_u16, acc=u32) >> add_u16_u32;
            op add(a_u32, b_u32, acc=u64) >> add_u32_u64;
            op add(a_i4, b_i4, acc=i8) >> add_i4_i8;
            op add(a_i2, b_i2, acc=i8) >> add_i2_i8;
            op add(a_i1, b_i1, acc=i8) >> add_i1_i8;
            op add(a_u4, b_u4, acc=u8) >> add_u4_u8;
            op add(a_u2, b_u2, acc=u8) >> add_u2_u8;
            op add(a_u1, b_u1, acc=u8) >> add_u1_u8;

            op mul(a_i8, b_i8, acc=i16) >> mul_i8_i16;
            op mul(a_i16, b_i16, acc=i32) >> mul_i16_i32;
            op mul(a_i32, b_i32, acc=i64) >> mul_i32_i64;
            op mul(a_u8, b_u8, acc=u16) >> mul_u8_u16;
            op mul(a_u16, b_u16, acc=u32) >> mul_u16_u32;
            op mul(a_u32, b_u32, acc=u64) >> mul_u32_u64;
            op mul(a_i4, b_i4, acc=i8) >> mul_i4_i8;
            op mul(a_i2, b_i2, acc=i8) >> mul_i2_i8;
            op mul(a_i1, b_i1, acc=i8) >> mul_i1_i8;
            op mul(a_u4, b_u4, acc=u8) >> mul_u4_u8;
            op mul(a_u2, b_u2, acc=u8) >> mul_u2_u8;
            op mul(a_u1, b_u1, acc=u8) >> mul_u1_u8;

            op abs(a_i8, acc=i16) >> abs_i8_i16;
            op abs(a_i16, acc=i32) >> abs_i16_i32;
            op abs(a_i32, acc=i64) >> abs_i32_i64;
            op abs(a_i4, acc=i8) >> abs_i4_i8;
            op abs(a_i2, acc=i8) >> abs_i2_i8;
            op abs(a_i1, acc=i8) >> abs_i1_i8;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;
    exec.step()?;

    for name in OUTPUTS {
        let value: TensorValue = exec.fetch(name)?;
        print_tensor_value(name, value);
    }

    Ok(())
}
