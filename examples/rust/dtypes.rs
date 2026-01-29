use openinfer::{graph, Device, FormatValue, ModelLoader, Simulator, TensorValue};
use std::path::Path;

mod util;
use util::select_device;

fn device_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::CpuAvx => "cpu-avx",
        Device::CpuAvx2 => "cpu-avx2",
        Device::Vulkan => "vulkan",
    }
}

fn try_graph(label: &str, model: &ModelLoader, graph: &openinfer::Graph, device: Device) {
    if !device.is_supported() {
        openinfer::trace!(
            "{}: device {} not supported in this build",
            label,
            device_name(device)
        );
        return;
    }
    match Simulator::new(model, graph, device).and_then(|sim| sim.make_executor()) {
        Ok(_) => openinfer::trace!("{}: ok", label),
        Err(err) => openinfer::trace!("{}: error: {}", label, err),
    }
}

fn format_full<T: FormatValue>(data: &[T]) -> String {
    if data.is_empty() {
        return "{}".to_string();
    }
    let joined = data
        .iter()
        .map(FormatValue::format_value)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{}}}", joined)
}

fn sign_extend(raw: u8, width: u8) -> i8 {
    let shift = 8 - width;
    ((raw << shift) as i8) >> shift
}

fn unpack_unsigned(bits: &[u8], bits_per: u8, count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    let mask = (1u16 << bits_per) - 1;
    for idx in 0..count {
        let bit_index = idx * bits_per as usize;
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let mut raw = (bits[byte_index] as u16) >> shift;
        if shift + bits_per as usize > 8 {
            raw |= (bits[byte_index + 1] as u16) << (8 - shift);
        }
        raw &= mask;
        out.push(raw as u8);
    }
    out
}

fn unpack_signed(bits: &[u8], bits_per: u8, count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    let mask = (1u16 << bits_per) - 1;
    for idx in 0..count {
        let bit_index = idx * bits_per as usize;
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let mut raw = (bits[byte_index] as u16) >> shift;
        if shift + bits_per as usize > 8 {
            raw |= (bits[byte_index + 1] as u16) << (8 - shift);
        }
        let value = sign_extend((raw & mask) as u8, bits_per);
        out.push(value);
    }
    out
}

fn unpack_t1(bits: &[u8], count: usize) -> Vec<i8> {
    unpack_unsigned(bits, 1, count)
        .into_iter()
        .map(|v| if v == 0 { -1 } else { 1 })
        .collect()
}

fn print_tensor(model: &ModelLoader, name: &str) -> anyhow::Result<()> {
    let tensor = model.load_tensor(name)?;
    openinfer::trace!(
        "{}: dtype={:?} shape={:?}",
        name,
        tensor.dtype(),
        tensor.shape()
    );
    match tensor {
        TensorValue::I8(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::I16(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::I32(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::I64(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::U8(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::U16(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::U32(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::U64(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::F16(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::BF16(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::F8(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::F32(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::F64(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::Bool(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::Bitset(t) => openinfer::trace!("  values={}", format_full(&t.data)),
        TensorValue::I4(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_signed(&bytes, 4, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::I2(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_signed(&bytes, 2, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::I1(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_signed(&bytes, 1, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::U4(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_unsigned(&bytes, 4, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::U2(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_unsigned(&bytes, 2, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::U1(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_unsigned(&bytes, 1, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::T2(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_signed(&bytes, 2, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
        TensorValue::T1(t) => {
            let bytes: Vec<u8> = t.data.iter().map(|v| v.bits).collect();
            let unpacked = unpack_t1(&bytes, t.numel());
            openinfer::trace!("  packed_bytes={}", format_full(&bytes));
            openinfer::trace!("  values={}", format_full(&unpacked));
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/dtypes_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let graph_universal = graph! {
        constant {
            a_f64: f64[N];
            b_f32: f32[N];
            c_i64: i64[N];
            d_i32: i32[N];
            e_i16: i16[N];
            f_i8: i8[N];
            g_u64: u64[N];
            h_u32: u32[N];
            i_u16: u16[N];
            j_u8: u8[N];
            k_bool: bool[N];
        }

        block entry {
            return;
        }
    };

    let graph_special = graph! {
        constant {
            l_f16: f16[N];
            m_bf16: bf16[N];
            n_f8: f8[N];
            o_i4: i4[N];
            p_i2: i2[N];
            q_i1: i1[N];
            r_u4: u4[N];
            s_u2: u2[N];
            t_u1: u1[N];
            u_t2: t2[N];
            v_t1: t1[N];
        }

        block entry {
            return;
        }
    };

    let device = select_device()?;
    let device_label = device_name(device);

    try_graph(&format!("{} universal graph", device_label), &model, &graph_universal, device);
    try_graph(&format!("{} special graph", device_label), &model, &graph_special, device);

    let tensor_names = [
        "a_f64", "b_f32", "c_i64", "d_i32", "e_i16", "f_i8", "g_u64", "h_u32", "i_u16", "j_u8",
        "k_bool", "l_f16", "m_bf16", "n_f8", "o_i4", "p_i2", "q_i1", "r_u4", "s_u2", "t_u1",
        "u_t2", "v_t1",
    ];
    for name in tensor_names {
        print_tensor(&model, name)?;
    }

    Ok(())
}
