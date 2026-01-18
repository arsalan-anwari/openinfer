use openinfer::{graph, Device, ModelLoader, Simulator};
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
        println!("{}: device {} not supported in this build", label, device_name(device));
        return;
    }
    match Simulator::new(model, graph, device).and_then(|sim| sim.make_executor()) {
        Ok(_) => println!("{}: ok", label),
        Err(err) => println!("{}: error: {}", label, err),
    }
}

fn main() -> anyhow::Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/dtypes_model.oinf");
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
        }

        block entry {
            return;
        }
    };

    let device = select_device()?;
    let device_label = device_name(device);

    if device == Device::Vulkan {
        #[cfg(feature = "vulkan")]
        {
            let features = openinfer::vulkan_features()?;
            println!(
                "vulkan features: i64/u64={}, f64={}, f16={}",
                features.supports_i64, features.supports_f64, features.supports_f16
            );
        }
        #[cfg(not(feature = "vulkan"))]
        {
            println!("vulkan features: unavailable (vulkan feature disabled)");
        }
    }

    try_graph(&format!("{} universal graph", device_label), &model, &graph_universal, device);
    try_graph(&format!("{} special graph", device_label), &model, &graph_special, device);

    Ok(())
}
