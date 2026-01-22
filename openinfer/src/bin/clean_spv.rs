use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use anyhow::{Context, Result};

const BROADCAST_SPV_DIR: &str = "src/backend/vulkan/broadcast/bin";
const MANIFEST_PATH: &str = "src/ops/vulkan/shaders.json";

fn main() -> Result<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.parent().unwrap_or(&manifest_dir).join("target"));

    let spv_dirs = collect_spv_dirs(&manifest_dir)?;
    let removed_spv = remove_spv_files(&spv_dirs)?;
    let removed_embedded = remove_embedded_artifacts(&target_dir)?;

    println!(
        "Removed {} SPIR-V files and {} embedded artifacts.",
        removed_spv, removed_embedded
    );

    let status = run_cargo_clean()?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

fn collect_spv_dirs(manifest_dir: &Path) -> Result<Vec<PathBuf>> {
    let manifest_path = manifest_dir.join(MANIFEST_PATH);
    let contents = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;

    let mut dirs = Vec::new();
    if let Some(ops) = manifest.get("ops").and_then(|v| v.as_object()) {
        for entry in ops.values() {
            if let Some(spv_dir) = entry.get("spv_dir").and_then(|v| v.as_str()) {
                dirs.push(manifest_dir.join(spv_dir));
            }
        }
    }
    dirs.push(manifest_dir.join(BROADCAST_SPV_DIR));
    Ok(dirs)
}

fn remove_spv_files(spv_dirs: &[PathBuf]) -> Result<usize> {
    let mut removed = 0;
    for spv_dir in spv_dirs {
        if !spv_dir.exists() {
            continue;
        }
        for entry in fs::read_dir(spv_dir)
            .with_context(|| format!("failed to read {}", spv_dir.display()))?
        {
            let path = entry?.path();
            if path.extension().map(|ext| ext == "spv").unwrap_or(false) {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

fn remove_embedded_artifacts(target_dir: &Path) -> Result<usize> {
    if !target_dir.exists() {
        return Ok(0);
    }
    let mut removed = 0;
    for entry in fs::read_dir(target_dir)
        .with_context(|| format!("failed to read {}", target_dir.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            removed += remove_embedded_artifacts(&path)?;
        } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            if name == "vulkan_spirv.rs" || name == "vulkan_spirv.d" {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

fn run_cargo_clean() -> Result<ExitStatus> {
    let mut cmd = Command::new("cargo");
    cmd.arg("clean");
    for arg in env::args().skip(1) {
        cmd.arg(arg);
    }
    cmd.status().with_context(|| "failed to run cargo clean")
}
