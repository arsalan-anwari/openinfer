use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::backend::{OpShaderInfo, ShaderRegistry};
use crate::graph::OpKind;
use crate::tensor::DType;

pub mod runtime;
pub use runtime::{storage_size_bytes, VulkanBufferInner, VulkanRuntime};

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
    spv_by_dtype: HashMap<String, String>,
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
            let op_kind = match name.as_str() {
                "abs" => OpKind::Abs,
                "add" => OpKind::Add,
                "mul" => OpKind::Mul,
                _ => continue,
            };
            let mut spv_paths_by_dtype = HashMap::new();
            for (dtype_name, spv_path) in entry.spv_by_dtype {
                if let Ok(dtype) = crate::tensor::DType::from_ident(&dtype_name) {
                    spv_paths_by_dtype.insert(dtype, spv_path);
                }
            }
            let spv_by_dtype = embedded_spirv_for_op(op_kind);
            ops.insert(
                name,
                Arc::new(OpShaderInfo {
                    path: entry.path,
                    spv_paths_by_dtype,
                    push_constants_size: entry.push_constants_size,
                    settings: entry.settings,
                    spv_by_dtype,
                }),
            );
        }
        Ok(Self { ops })
    }

    #[allow(unused)]
    pub fn load_default() -> Result<Self> {
        Self::load_from_file(Self::default_manifest_path())
    }

    #[allow(unused)]
    pub fn default_manifest_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ops/vulkan/shaders.json")
    }
}

impl ShaderRegistry for VulkanShaderRegistry {
    fn shader_for_op(&self, op: OpKind) -> Option<Arc<OpShaderInfo>> {
        self.ops.get(op.as_str()).cloned()
    }
}

fn embedded_spirv_for_op(op: OpKind) -> HashMap<DType, &'static [u8]> {
    use crate::tensor::DType;
    let mut map = HashMap::new();
    match op {
        OpKind::Abs => {
            map.insert(DType::I8, &include_bytes!("../../ops/vulkan/abs/bin/abs_i8.spv")[..]);
            map.insert(DType::I16, &include_bytes!("../../ops/vulkan/abs/bin/abs_i16.spv")[..]);
            map.insert(DType::F32, &include_bytes!("../../ops/vulkan/abs/bin/abs_f32.spv")[..]);
            map.insert(DType::I32, &include_bytes!("../../ops/vulkan/abs/bin/abs_i32.spv")[..]);
            map.insert(DType::I64, &include_bytes!("../../ops/vulkan/abs/bin/abs_i64.spv")[..]);
        }
        OpKind::Add => {
            map.insert(DType::I8, &include_bytes!("../../ops/vulkan/add/bin/add_i8.spv")[..]);
            map.insert(DType::I16, &include_bytes!("../../ops/vulkan/add/bin/add_i16.spv")[..]);
            map.insert(DType::F32, &include_bytes!("../../ops/vulkan/add/bin/add_f32.spv")[..]);
            map.insert(DType::Bool, &include_bytes!("../../ops/vulkan/add/bin/add_bool.spv")[..]);
            map.insert(DType::U8, &include_bytes!("../../ops/vulkan/add/bin/add_u8.spv")[..]);
            map.insert(DType::U16, &include_bytes!("../../ops/vulkan/add/bin/add_u16.spv")[..]);
            map.insert(DType::I32, &include_bytes!("../../ops/vulkan/add/bin/add_i32.spv")[..]);
            map.insert(DType::U32, &include_bytes!("../../ops/vulkan/add/bin/add_u32.spv")[..]);
            map.insert(DType::I64, &include_bytes!("../../ops/vulkan/add/bin/add_i64.spv")[..]);
            map.insert(DType::U64, &include_bytes!("../../ops/vulkan/add/bin/add_u64.spv")[..]);
        }
        OpKind::Mul => {
            map.insert(DType::I8, &include_bytes!("../../ops/vulkan/mul/bin/mul_i8.spv")[..]);
            map.insert(DType::I16, &include_bytes!("../../ops/vulkan/mul/bin/mul_i16.spv")[..]);
            map.insert(DType::F32, &include_bytes!("../../ops/vulkan/mul/bin/mul_f32.spv")[..]);
            map.insert(DType::Bool, &include_bytes!("../../ops/vulkan/mul/bin/mul_bool.spv")[..]);
            map.insert(DType::U8, &include_bytes!("../../ops/vulkan/mul/bin/mul_u8.spv")[..]);
            map.insert(DType::U16, &include_bytes!("../../ops/vulkan/mul/bin/mul_u16.spv")[..]);
            map.insert(DType::I32, &include_bytes!("../../ops/vulkan/mul/bin/mul_i32.spv")[..]);
            map.insert(DType::U32, &include_bytes!("../../ops/vulkan/mul/bin/mul_u32.spv")[..]);
            map.insert(DType::I64, &include_bytes!("../../ops/vulkan/mul/bin/mul_i64.spv")[..]);
            map.insert(DType::U64, &include_bytes!("../../ops/vulkan/mul/bin/mul_u64.spv")[..]);
        }
    }
    map
}
