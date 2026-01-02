use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct VulkanShaderManifest {
    ops: HashMap<String, VulkanShaderEntry>,
}

#[derive(Debug, Deserialize)]
struct VulkanShaderEntry {
    path: String,
    spv_by_dtype: HashMap<String, String>,
}

fn main() -> Result<()> {
    if env::var("CARGO_FEATURE_VULKAN").is_err() {
        return Ok(());
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
    let manifest_path = manifest_dir.join("src/ops/vulkan/shaders.json");
    println!("cargo:rerun-if-changed={}", manifest_path.display());

    let contents = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: VulkanShaderManifest = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;

    for (op_name, entry) in manifest.ops {
        let src_path = manifest_dir.join(&entry.path);
        println!("cargo:rerun-if-changed={}", src_path.display());

        for (dtype, spv_path_str) in entry.spv_by_dtype {
            let spv_path = manifest_dir.join(&spv_path_str);
            let should_compile = needs_rebuild(&src_path, &spv_path)?;
            if should_compile {
                let entry_point = format!("{}_{}", op_name, dtype);
                compile_slang(&src_path, &spv_path, &entry_point)?;
            }
        }
    }

    Ok(())
}

fn needs_rebuild(src: &Path, spv: &Path) -> Result<bool> {
    if !spv.exists() {
        return Ok(true);
    }
    let src_meta = fs::metadata(src)
        .with_context(|| format!("failed to stat {}", src.display()))?;
    let spv_meta = fs::metadata(spv)
        .with_context(|| format!("failed to stat {}", spv.display()))?;
    let src_time = src_meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let spv_time = spv_meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    Ok(src_time > spv_time)
}

fn compile_slang(src: &Path, spv: &Path, entry_point: &str) -> Result<()> {
    if let Some(parent) = spv.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_string());
    let status = Command::new(&slangc)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("sm_6_2")
        .arg("-stage")
        .arg("compute")
        .arg("-entry")
        .arg(entry_point)
        .arg("-o")
        .arg(spv)
        .arg(src)
        .status()
        .with_context(|| {
            format!(
                "failed to spawn slangc for {} (set SLANGC or add slangc to PATH)",
                src.display()
            )
        })?;

    if !status.success() {
        return Err(anyhow!("slangc failed for {}", src.display()));
    }

    Ok(())
}
