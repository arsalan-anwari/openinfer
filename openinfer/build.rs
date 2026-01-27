use std::env;
use std::path::PathBuf;

mod generator;

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
    let op_types = manifest_dir.join("src/graph/types.rs");
    println!("cargo:rerun-if-changed={}", op_defs.display());
    println!("cargo:rerun-if-changed={}", op_dtypes.display());
    println!("cargo:rerun-if-changed={}", op_types.display());
    if let Err(err) = generator::op_schema::generate_cpu_kernels(&manifest_dir) {
        eprintln!("build.rs: failed to generate cpu kernels: {err}");
    }
    if let Err(err) = generator::settings::apply_settings(&manifest_dir) {
        eprintln!("build.rs: failed to apply settings: {err}");
    }
    if let Err(err) = generator::vulkan_spv::generate_spv_map(&manifest_dir) {
        eprintln!("build.rs: failed to generate embedded spv map: {err}");
        if let Err(write_err) = generator::vulkan_spv::write_empty_map() {
            eprintln!("build.rs: failed to write empty spv map: {write_err}");
        }
    }
}
