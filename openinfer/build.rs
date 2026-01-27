use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(err) = apply_settings() {
        eprintln!("build.rs: failed to apply settings: {err}");
    }
    if let Err(err) = generate_spv_map() {
        eprintln!("build.rs: failed to generate embedded spv map: {err}");
        if let Err(write_err) = write_empty_map() {
            eprintln!("build.rs: failed to write empty spv map: {write_err}");
        }
    }
}

fn apply_settings() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let settings_path = manifest_dir.join("../settings.json");
    println!("cargo:rerun-if-changed={}", settings_path.display());
    let max_dims = read_settings_max_dims(&settings_path).unwrap_or(8);
    println!("cargo:rustc-env=OPENINFER_VK_MAX_DIMS={}", max_dims);
    write_shader_config(&manifest_dir, max_dims)?;
    write_rust_config(max_dims)?;
    Ok(())
}

fn read_settings_max_dims(path: &Path) -> Option<usize> {
    let contents = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    value
        .get("openinfer")
        .and_then(|v| v.get("vulkan"))
        .and_then(|v| v.get("max_tensor_rank"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

fn write_shader_config(manifest_dir: &Path, max_dims: usize) -> Result<(), Box<dyn std::error::Error>> {
    let shader_config = manifest_dir.join("src/ops/vulkan/shaders/generated_config.slang");
    let contents = format!(
        "#ifndef OPENINFER_GENERATED_CONFIG\n#define OPENINFER_GENERATED_CONFIG 1\n#define OPENINFER_VK_MAX_DIMS {}\n#endif\n",
        max_dims
    );
    fs::write(shader_config, contents)?;
    Ok(())
}

fn write_rust_config(max_dims: usize) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let out_file = out_dir.join("vulkan_config.rs");
    let contents = format!("pub const MAX_DIMS: usize = {max_dims};\n");
    fs::write(out_file, contents)?;
    Ok(())
}

fn generate_spv_map() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let shaders_json = manifest_dir.join("src/ops/vulkan/shaders.json");
    println!("cargo:rerun-if-changed={}", shaders_json.display());
    let contents = fs::read_to_string(&shaders_json)?;
    let value: serde_json::Value = serde_json::from_str(&contents)?;
    let ops = value
        .get("ops")
        .and_then(|ops| ops.as_object())
        .ok_or("shaders.json missing ops object")?;

    let mut entries: Vec<(String, PathBuf)> = Vec::new();

    for (op, config) in ops {
        let shader_dir = config
            .get("shader_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{op} missing shader_dir"))?;
        let spv_dir = config
            .get("spv_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{op} missing spv_dir"))?;
        let shader_files = config
            .get("shader_files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{op} missing shader_files"))?;

        let shader_dir = manifest_dir.join(shader_dir);
        let spv_dir = manifest_dir.join(spv_dir);

        for file in shader_files {
            let file = file
                .as_str()
                .ok_or_else(|| format!("{op} shader_files must be strings"))?;
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

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
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

fn write_empty_map() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let out_file = out_dir.join("spv_embedded.rs");
    let output = "pub fn embedded_spv(_: &str) -> Option<&'static [u8]> { None }\n";
    fs::write(out_file, output)?;
    Ok(())
}

fn parse_entrypoints(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
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
