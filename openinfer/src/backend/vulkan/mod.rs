use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::backend::{OpShaderInfo, ShaderRegistry};
use crate::graph::OpKind;

pub mod runtime;
pub use runtime::{storage_size_bytes, VulkanBufferInner, VulkanRuntime};

mod embedded_spirv {
    include!(concat!(env!("OUT_DIR"), "/vulkan_spirv.rs"));
}

#[derive(Debug, Default)]
pub struct VulkanShaderRegistry {
    ops: HashMap<String, Arc<OpShaderInfo>>,
}

#[derive(Debug, Deserialize)]
struct VulkanShaderManifest {
    ops: HashMap<String, VulkanShaderEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct VulkanShaderEntry {
    path: String,
    spv_dir: String,
    push_constants_size: usize,
    #[serde(default)]
    settings: HashMap<String, Value>,
}

impl VulkanShaderRegistry {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read vulkan shader manifest at {}", path.display()))?;
        let manifest: VulkanShaderManifest =
            serde_json::from_str(&contents).with_context(|| {
                format!("failed to parse vulkan shader manifest at {}", path.display())
            })?;
        let mut ops = HashMap::new();
        for (name, entry) in manifest.ops {
            if !matches!(name.as_str(), "abs" | "add" | "mul") {
                continue;
            }
            let spv_by_target = embedded_spirv::embedded_spirv_for_op(name.as_str());
            ops.insert(
                name,
                Arc::new(OpShaderInfo {
                    path: entry.path,
                    spv_dir: entry.spv_dir,
                    push_constants_size: entry.push_constants_size,
                    settings: entry.settings,
                    spv_by_target,
                }),
            );
        }
        Ok(Self { ops })
    }

    pub fn load_default() -> Result<Self> {
        Self::load_from_file(Self::default_manifest_path())
    }

    pub fn default_manifest_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ops/vulkan/shaders.json")
    }
}

impl ShaderRegistry for VulkanShaderRegistry {
    fn shader_for_op(&self, op: OpKind) -> Option<Arc<OpShaderInfo>> {
        self.ops.get(op.as_str()).cloned()
    }
}
