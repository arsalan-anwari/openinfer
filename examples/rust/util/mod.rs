use anyhow::{anyhow, Result};
use openinfer::Device;
use std::env;

fn default_device() -> Device {
    if cfg!(feature = "avx2") {
        Device::CpuAvx2
    } else if cfg!(feature = "avx") {
        Device::CpuAvx
    } else {
        Device::Cpu
    }
}

fn parse_example_target() -> Result<Option<String>> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--target=") {
            return Ok(Some(value.to_string()));
        }
        if arg == "--target" {
            return match args.next() {
                Some(value) => Ok(Some(value)),
                None => Err(anyhow!(
                    "--target requires a value: cpu|avx|avx2|vulkan"
                )),
            };
        }
    }
    Ok(None)
}

pub fn select_device() -> Result<Device> {
    let target = match parse_example_target()? {
        Some(target) => target,
        None => return Ok(default_device()),
    };

    match target.as_str() {
        "cpu" => Ok(Device::Cpu),
        "avx" => Ok(Device::CpuAvx),
        "avx2" => Ok(Device::CpuAvx2),
        "vulkan" => Ok(Device::Vulkan),
        _ => Err(anyhow!(
            "unknown --target value '{}'; expected cpu|avx|avx2|vulkan",
            target
        )),
    }
}
