use std::env;
use std::fs;
use std::path::PathBuf;

#[path = "generator/settings.rs"]
mod settings;
#[path = "generator/vulkan_spv.rs"]
mod vulkan_spv;

fn main() {
    let manifest_dir = match env::var("CARGO_MANIFEST_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(err) => {
            eprintln!("build.rs: missing CARGO_MANIFEST_DIR: {err}");
            return;
        }
    };
    let op_defs = manifest_dir.join("src/registry/op_defs.rs");
    let op_dtypes = manifest_dir.join("src/registry/op_dtypes.rs");
    let op_dtypes_dir = manifest_dir.join("src/registry/op_dtypes");
    let op_types = manifest_dir.join("src/graph/types.rs");
    println!("cargo:rerun-if-changed={}", op_defs.display());
    println!("cargo:rerun-if-changed={}", op_dtypes.display());
    if let Ok(entries) = fs::read_dir(&op_dtypes_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    println!("cargo:rerun-if-changed={}", op_types.display());
    if let Err(err) = settings::apply_settings(&manifest_dir) {
        eprintln!("build.rs: failed to apply settings: {err}");
    }
    if let Err(err) = vulkan_spv::generate_spv_map(&manifest_dir) {
        eprintln!("build.rs: failed to generate embedded spv map: {err}");
        if let Err(write_err) = vulkan_spv::write_empty_map() {
            eprintln!("build.rs: failed to write empty spv map: {write_err}");
        }
    }
}
