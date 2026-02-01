use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use ash::vk;
use serde_json::Value;

fn main() -> Result<()> {
    let ops_filter = parse_ops_filter()?;
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let settings_path = workspace_root.join("settings.json");
    let openinfer_dir = workspace_root.join("openinfer");
    let ops_json = workspace_root.join("ops.json");
    let contents = fs::read_to_string(&ops_json).with_context(|| format!("read {}", ops_json.display()))?;
    let value: Value = serde_json::from_str(&contents)?;
    let ops = value
        .get("ops")
        .and_then(|ops| ops.as_array())
        .ok_or_else(|| anyhow!("ops.json missing ops array"))?;

    let mut planned = Vec::new();
    for op in ops {
        let op_name = op
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("ops.json op missing name"))?;
        if let Some(filter) = &ops_filter {
            if !filter.contains(op_name) {
                continue;
            }
        }
        let vulkan = op
            .get("devices")
            .and_then(|v| v.as_object())
            .and_then(|v| v.get("vulkan"))
            .and_then(|v| v.as_object());
        let Some(vulkan) = vulkan else {
            continue;
        };
        let shader_dir = vulkan
            .get("shader_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{op_name} missing shader_dir"))?;
        let spv_dir = vulkan
            .get("spv_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{op_name} missing spv_dir"))?;
        let shader_files = vulkan
            .get("shader_files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("{op_name} missing shader_files"))?;

        let shader_dir = openinfer_dir.join(shader_dir);
        let spv_dir = openinfer_dir.join(spv_dir);
        fs::create_dir_all(&spv_dir)?;

        let include_dir = openinfer_dir.join("src/ops/vulkan/shaders");

        for file in shader_files {
            let file = file
                .as_str()
                .ok_or_else(|| anyhow!("{op_name} shader_files must be strings"))?;
            let shader_path = shader_dir.join(file);
            let entrypoints = parse_entrypoints(&shader_path)?;
            for entry in entrypoints {
                let (has_f64, has_i64, has_u64) =
                    resolve_feature_flags(&settings_path)?;
                if should_skip_entry(&entry, has_f64, has_i64, has_u64) {
                    continue;
                }
                let spv_path = spv_dir.join(format!("{}.spv", entry));
                let spv_display = spv_path
                    .strip_prefix(&openinfer_dir)
                    .unwrap_or(&spv_path)
                    .display()
                    .to_string();
                planned.push(PlannedCompile {
                    op_name: op_name.to_string(),
                    shader_path: shader_path.clone(),
                    shader_dir: shader_dir.clone(),
                    include_dir: include_dir.clone(),
                    entry,
                    spv_path,
                    spv_display,
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
            .arg("-Wno-30081")
            .arg("-o")
            .arg(&item.spv_path)
            .arg("-I")
            .arg(&item.shader_dir)
            .arg("-I")
            .arg(&item.include_dir)
            .arg("-D")
            .arg(format!("HAS_F64={}", item.has_f64))
            .arg("-D")
            .arg(format!("HAS_I64={}", item.has_i64))
            .arg("-D")
            .arg(format!("HAS_U64={}", item.has_u64));
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

fn parse_ops_filter() -> Result<Option<HashSet<String>>> {
    let mut args = std::env::args().skip(1);
    let mut ops: Option<HashSet<String>> = None;
    while let Some(arg) = args.next() {
        if arg == "--ops" {
            let value = args
                .next()
                .ok_or_else(|| anyhow!("--ops requires a value"))?;
            ops = Some(parse_ops_value(&value)?);
        } else if let Some(value) = arg.strip_prefix("--ops=") {
            ops = Some(parse_ops_value(value)?);
        } else {
            return Err(anyhow!("unknown argument: {}", arg));
        }
    }
    Ok(ops)
}

fn parse_ops_value(value: &str) -> Result<HashSet<String>> {
    let mut set = HashSet::new();
    for raw in value.split(',') {
        let op = raw.trim();
        if op.is_empty() {
            continue;
        }
        set.insert(op.to_string());
    }
    if set.is_empty() {
        return Err(anyhow!("--ops must contain at least one op"));
    }
    Ok(set)
}

struct PlannedCompile {
    op_name: String,
    shader_path: PathBuf,
    shader_dir: PathBuf,
    include_dir: PathBuf,
    entry: String,
    spv_path: PathBuf,
    spv_display: String,
    has_f64: u32,
    has_i64: u32,
    has_u64: u32,
}

fn resolve_feature_flags(settings_path: &Path) -> Result<(u32, u32, u32)> {
    if let Some(flags) = read_settings_flags(settings_path)? {
        return Ok(flags);
    }
    if let Some(caps) = probe_vulkan_caps() {
        return Ok((
            if caps.float64 { 1 } else { 0 },
            if caps.int64 { 1 } else { 0 },
            if caps.int64 { 1 } else { 0 },
        ));
    }
    Ok((0, 0, 0))
}

fn read_settings_flags(settings_path: &Path) -> Result<Option<(u32, u32, u32)>> {
    if !settings_path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(settings_path)
        .with_context(|| format!("read {}", settings_path.display()))?;
    let value: Value = serde_json::from_str(&contents)?;
    let vulkan = value
        .get("openinfer")
        .and_then(|v| v.get("vulkan"))
        .and_then(|v| v.as_object());
    let Some(vulkan) = vulkan else {
        return Ok(None);
    };

    let has_f64 = vulkan.get("has_f64").and_then(|v| v.as_bool());
    let has_i64 = vulkan.get("has_i64").and_then(|v| v.as_bool());
    let has_u64 = vulkan.get("has_u64").and_then(|v| v.as_bool());
    if  has_f64.is_none() && has_i64.is_none() && has_u64.is_none() {
        return Ok(None);
    }
    Ok(Some((
        if has_f64.unwrap_or(false) { 1 } else { 0 },
        if has_i64.unwrap_or(false) { 1 } else { 0 },
        if has_u64.unwrap_or(false) { 1 } else { 0 },
    )))
}

#[derive(Clone, Copy)]
struct VulkanCaps {
    int64: bool,
    float64: bool
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
    
    Some(VulkanCaps {
        int64: device_features.shader_int64 == vk::TRUE,
        float64: device_features.shader_float64 == vk::TRUE
    })
}

fn should_skip_entry(entry: &str, has_f64: u32, has_i64: u32, has_u64: u32) -> bool {
    let lower = entry.to_ascii_lowercase();

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
    let lines: Vec<&str> = contents.lines().collect();
    let entry_macros = parse_entrypoint_macros(&lines);

    let mut entrypoints = Vec::new();
    let mut expect_entry = false;
    let mut in_macro = false;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("#define ") {
            in_macro = trimmed.ends_with('\\');
            expect_entry = false;
            continue;
        }
        if in_macro {
            if !trimmed.ends_with('\\') {
                in_macro = false;
            }
            continue;
        }

        if trimmed.contains("[shader(\"compute\")]") {
            expect_entry = true;
            continue;
        }

        if expect_entry {
            if trimmed.starts_with('#') {
                continue;
            }
            if let Some(name) = parse_void_name(trimmed) {
                entrypoints.push(name);
                expect_entry = false;
                continue;
            }
            if !trimmed.is_empty() && !trimmed.starts_with('[') {
                expect_entry = false;
            }
        }

        if trimmed.starts_with('#') || trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }
        if let Some(name) = parse_macro_invocation(trimmed, &entry_macros) {
            entrypoints.push(name);
        }
    }

    Ok(entrypoints)
}

fn parse_entrypoint_macros(lines: &[&str]) -> std::collections::HashMap<String, usize> {
    let mut macros = std::collections::HashMap::new();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx].trim();
        if !line.starts_with("#define ") {
            idx += 1;
            continue;
        }
        let after_define = line.trim_start_matches("#define").trim();
        let open = after_define.find('(');
        let close = after_define.find(')');
        let (Some(open), Some(close)) = (open, close) else {
            idx += 1;
            continue;
        };
        let name = after_define[..open].trim();
        if name.is_empty() {
            idx += 1;
            continue;
        }
        let params_raw = &after_define[open + 1..close];
        let params: Vec<String> = params_raw
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if params.is_empty() {
            idx += 1;
            continue;
        }

        let mut has_shader = false;
        let mut entry_param: Option<usize> = None;
        let mut macro_idx = idx;
        loop {
            let body_line = lines[macro_idx].trim();
            if body_line.contains("[shader(\"compute\")]") {
                has_shader = true;
            }
            for (param_idx, param) in params.iter().enumerate() {
                let needle = format!("void {}", param);
                if body_line.contains(&needle) {
                    entry_param = Some(param_idx);
                }
            }
            if !body_line.ends_with('\\') {
                break;
            }
            macro_idx += 1;
            if macro_idx >= lines.len() {
                break;
            }
        }
        if has_shader {
            if let Some(param_idx) = entry_param {
                macros.insert(name.to_string(), param_idx);
            }
        }
        idx = macro_idx.saturating_add(1);
    }
    macros
}

fn parse_macro_invocation(
    line: &str,
    entry_macros: &std::collections::HashMap<String, usize>,
) -> Option<String> {
    let open = line.find('(')?;
    let close = line.rfind(')')?;
    if close <= open {
        return None;
    }
    let macro_name = line[..open].trim();
    let param_idx = *entry_macros.get(macro_name)?;
    let args_raw = &line[open + 1..close];
    let args: Vec<&str> = args_raw.split(',').map(|s| s.trim()).collect();
    let entry = args.get(param_idx)?.trim();
    if entry.is_empty() {
        None
    } else {
        Some(entry.to_string())
    }
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
    let _ = write!(io::stdout(), "\r{}{}", line, pad);
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

