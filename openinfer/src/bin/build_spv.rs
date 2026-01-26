use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::io::{self, Write};

use anyhow::{anyhow, Context, Result};
use ash::vk;
use serde_json::Value;

fn main() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shaders_json = manifest_dir.join("src/ops/vulkan/shaders.json");
    let contents =
        fs::read_to_string(&shaders_json).with_context(|| format!("read {}", shaders_json.display()))?;
    let value: Value = serde_json::from_str(&contents)?;
    let ops = value
        .get("ops")
        .and_then(|ops| ops.as_object())
        .ok_or_else(|| anyhow!("shaders.json missing ops object"))?;

    let mut planned = Vec::new();
    for (op, config) in ops {
        let shader_dir = config
            .get("shader_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{} missing shader_dir", op))?;
        let spv_dir = config
            .get("spv_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{} missing spv_dir", op))?;
        let shader_files = config
            .get("shader_files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("{} missing shader_files", op))?;

        let shader_dir = manifest_dir.join(shader_dir);
        let spv_dir = manifest_dir.join(spv_dir);
        fs::create_dir_all(&spv_dir)?;

        let include_dir = manifest_dir.join("src/ops/vulkan/shaders");

        for file in shader_files {
            let file = file
                .as_str()
                .ok_or_else(|| anyhow!("{} shader_files must be strings", op))?;
            let shader_path = shader_dir.join(file);
            let entrypoints = parse_entrypoints(&shader_path)?;
            for entry in entrypoints {
                let (has_f16, has_f64, has_i64, has_u64) = resolve_feature_flags()?;
                if should_skip_entry(&entry, has_f16, has_f64, has_i64, has_u64) {
                    continue;
                }
                let spv_path = spv_dir.join(format!("{}.spv", entry));
                let spv_display = spv_path
                    .strip_prefix(&manifest_dir)
                    .unwrap_or(&spv_path)
                    .display()
                    .to_string();
                planned.push(PlannedCompile {
                    op_name: op.to_string(),
                    shader_path: shader_path.clone(),
                    shader_dir: shader_dir.clone(),
                    include_dir: include_dir.clone(),
                    entry,
                    spv_path,
                    spv_display,
                    has_f16,
                    has_f64,
                    has_i64,
                    has_u64,
                });
            }
        }
    }

    let total = planned.len();
    let mut last_len = 0usize;
    for (idx, item) in planned.iter().enumerate() {
        let current = idx + 1;
        print_progress(current, total, item, &mut last_len);
        let mut cmd = Command::new("slangc");
        cmd.arg(&item.shader_path)
            .arg("-entry")
            .arg(&item.entry)
            .arg("-target")
            .arg("spirv")
            .arg("-o")
            .arg(&item.spv_path)
            .arg("-I")
            .arg(&item.shader_dir)
            .arg("-I")
            .arg(&item.include_dir)
            .arg("-D").arg(format!("HAS_F16={}", item.has_f16))
            .arg("-D").arg(format!("HAS_F64={}", item.has_f64))
            .arg("-D").arg(format!("HAS_I64={}", item.has_i64))
            .arg("-D").arg(format!("HAS_U64={}", item.has_u64));
        let status = cmd
            .status()
            .with_context(|| format!("run slangc for {}", item.shader_path.display()))?;
        if !status.success() {
            return Err(anyhow!(
                "slangc failed for {} entry {}",
                item.shader_path.display(),
                item.entry
            ));
        }
    }
    if total > 0 {
        println!();
    }

    Ok(())
}

struct PlannedCompile {
    op_name: String,
    shader_path: PathBuf,
    shader_dir: PathBuf,
    include_dir: PathBuf,
    entry: String,
    spv_path: PathBuf,
    spv_display: String,
    has_f16: u32,
    has_f64: u32,
    has_i64: u32,
    has_u64: u32,
}

fn env_flag(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .as_deref()
        .map(|v| if v == "1" { 1 } else { 0 })
}

fn resolve_feature_flags() -> Result<(u32, u32, u32, u32)> {
    let env_f16 = env_flag("HAS_F16");
    let env_f64 = env_flag("HAS_F64");
    let env_i64 = env_flag("HAS_I64");
    let env_u64 = env_flag("HAS_U64");
    if env_f16.is_some() || env_f64.is_some() || env_i64.is_some() || env_u64.is_some() {
        return Ok((
            env_f16.unwrap_or(0),
            env_f64.unwrap_or(0),
            env_i64.unwrap_or(0),
            env_u64.unwrap_or(0),
        ));
    }
    if let Some(caps) = probe_vulkan_caps() {
        return Ok((
            if caps.float16 { 1 } else { 0 },
            if caps.float64 { 1 } else { 0 },
            if caps.int64 { 1 } else { 0 },
            if caps.int64 { 1 } else { 0 },
        ));
    }
    Ok((0, 0, 0, 0))
}

#[derive(Clone, Copy)]
struct VulkanCaps {
    int64: bool,
    float64: bool,
    float16: bool,
}

fn probe_vulkan_caps() -> Option<VulkanCaps> {
    let entry = unsafe { ash::Entry::load().ok()? };
    let app_name = b"openinfer\0";
    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr() as *const i8,
        application_version: 0,
        p_engine_name: app_name.as_ptr() as *const i8,
        engine_version: 0,
        api_version: vk::make_api_version(0, 1, 1, 0),
        ..Default::default()
    };
    let instance_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    let instance = unsafe { entry.create_instance(&instance_info, None).ok()? };
    let physical_device = unsafe {
        instance
            .enumerate_physical_devices()
            .ok()?
            .get(0)
            .copied()
    }?;
    let device_features = unsafe { instance.get_physical_device_features(physical_device) };
    let mut float16_int8 = vk::PhysicalDeviceFloat16Int8FeaturesKHR::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default();
    unsafe {
        features2.p_next = &mut float16_int8 as *mut _ as *mut std::ffi::c_void;
        instance.get_physical_device_features2(physical_device, &mut features2);
        instance.destroy_instance(None);
    }
    Some(VulkanCaps {
        int64: device_features.shader_int64 == vk::TRUE,
        float64: device_features.shader_float64 == vk::TRUE,
        float16: float16_int8.shader_float16 == vk::TRUE,
    })
}

fn should_skip_entry(entry: &str, has_f16: u32, has_f64: u32, has_i64: u32, has_u64: u32) -> bool {
    let lower = entry.to_ascii_lowercase();
    if has_f16 == 0 && lower.contains("_native") {
        return true;
    }
    if has_f64 == 0 && (lower.contains("_f64") || lower.starts_with("add_f64")) {
        return true;
    }
    if has_i64 == 0 && (lower.contains("_i64") || lower.starts_with("add_i64")) {
        return true;
    }
    if has_u64 == 0 && (lower.contains("_u64") || lower.starts_with("add_u64")) {
        return true;
    }
    false
}

fn parse_entrypoints(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("read shader {}", path.display()))?;
    let mut entrypoints = Vec::new();
    let mut expect_entry = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.contains("[shader(\"compute\")]") {
            expect_entry = true;
            continue;
        }
        if expect_entry {
            if let Some(name) = parse_void_name(trimmed) {
                entrypoints.push(name);
                expect_entry = false;
            } else if !trimmed.is_empty() && !trimmed.starts_with('[') {
                expect_entry = false;
            }
        }
    }
    Ok(entrypoints)
}

fn parse_void_name(line: &str) -> Option<String> {
    let line = line.trim_start();
    if !line.starts_with("void ") {
        return None;
    }
    let rest = &line[5..];
    let name = rest
        .split('(')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())?;
    Some(name.to_string())
}

fn print_progress(current: usize, total: usize, item: &PlannedCompile, last_len: &mut usize) {
    let path = shorten_path(item.spv_display.clone(), 80);
    let line = format!(
        "[{}/{}] {}::{} --> {}",
        current,
        total,
        item.op_name,
        item.entry,
        path
    );
    let pad = if *last_len > line.len() {
        " ".repeat(*last_len - line.len())
    } else {
        String::new()
    };
    let _ = write!(
        io::stdout(),
        "\r{}{}",
        line,
        pad
    );
    let _ = io::stdout().flush();
    *last_len = line.len();
}

fn shorten_path(path: String, max_len: usize) -> String {
    if path.len() <= max_len {
        return path;
    }
    let keep = max_len.saturating_sub(3);
    let tail = path
        .chars()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("...{}", tail)
}
