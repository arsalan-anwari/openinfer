//! Generate embedded SPIR-V shader maps from `ops.json`.
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Generate a Rust map that embeds available SPIR-V binaries.
pub fn generate_spv_map(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let ops_json = manifest_dir.join("../ops.json");
    println!("cargo:rerun-if-changed={}", ops_json.display());
    let contents = fs::read_to_string(&ops_json)?;
    let value: serde_json::Value = serde_json::from_str(&contents)?;
    let ops = value
        .get("ops")
        .and_then(|ops| ops.as_array())
        .ok_or("ops.json missing ops array")?;

    let mut entries: Vec<(String, PathBuf)> = Vec::new();

    for op in ops {
        let op_name = op
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("ops.json op missing name")?;
        let vulkan = match op
            .get("devices")
            .and_then(|v| v.as_object())
            .and_then(|v| v.get("vulkan"))
            .and_then(|v| v.as_object())
        {
            Some(vulkan) => vulkan,
            None => continue,
        };
        let shader_dir = vulkan
            .get("shader_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{op_name} missing shader_dir"))?;
        let spv_dir = vulkan
            .get("spv_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{op_name} missing spv_dir"))?;
        let shader_files = vulkan
            .get("shader_files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{op_name} missing shader_files"))?;

        let shader_dir = manifest_dir.join(shader_dir);
        let spv_dir = manifest_dir.join(spv_dir);

        for file in shader_files {
            let file = file
                .as_str()
                .ok_or_else(|| format!("{op_name} shader_files must be strings"))?;
            let shader_path = shader_dir.join(file);
            println!("cargo:rerun-if-changed={}", shader_path.display());
            let entrypoints = parse_entrypoints(&shader_path)?;
            for entry in entrypoints {
                let spv_path = spv_dir.join(format!("{entry}.spv"));
                if spv_path.exists() {
                    entries.push((entry, spv_path));
                }
            }
        }
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let out_file = out_dir.join("spv_embedded.rs");
    let mut output = String::new();
    output.push_str("pub fn embedded_spv(name: &str) -> Option<&'static [u8]> {\n");
    output.push_str("    match name {\n");
    for (name, path) in entries {
        let path = path.to_string_lossy().replace('\\', "/");
        output.push_str(&format!(
            "        \"{name}\" => Some(include_bytes!(r#\"{path}\"#)),\n"
        ));
    }
    output.push_str("        _ => None,\n");
    output.push_str("    }\n");
    output.push_str("}\n");
    fs::write(out_file, output)?;
    Ok(())
}

/// Write an empty embedded SPIR-V map (for non-Vulkan builds).
pub fn write_empty_map() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let out_file = out_dir.join("spv_embedded.rs");
    let output = "pub fn embedded_spv(_: &str) -> Option<&'static [u8]> { None }\n";
    fs::write(out_file, output)?;
    Ok(())
}

fn parse_entrypoints(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
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
