use std::error::Error;
use std::fs;
use std::path::Path;

pub fn apply_settings(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let settings_path = manifest_dir.join("../settings.json");
    println!("cargo:rerun-if-changed={}", settings_path.display());
    let max_dims = read_settings_max_dims(&settings_path).unwrap_or(8);
    println!("cargo:rustc-env=OPENINFER_VK_MAX_DIMS={}", max_dims);
    write_shader_config(manifest_dir, max_dims)?;
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

fn write_shader_config(manifest_dir: &Path, max_dims: usize) -> Result<(), Box<dyn Error>> {
    let shader_config = manifest_dir.join("src/ops/vulkan/shaders/generated_config.slang");
    let contents = format!(
        "#ifndef OPENINFER_GENERATED_CONFIG\n#define OPENINFER_GENERATED_CONFIG 1\n#define OPENINFER_VK_MAX_DIMS {}\n#endif\n",
        max_dims
    );
    fs::write(shader_config, contents)?;
    Ok(())
}

fn write_rust_config(max_dims: usize) -> Result<(), Box<dyn Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let out_file = out_dir.join("vulkan_config.rs");
    let contents = format!("pub const MAX_DIMS: usize = {max_dims};\n");
    fs::write(out_file, contents)?;
    Ok(())
}
