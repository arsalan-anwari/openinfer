use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

fn main() -> Result<()> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let openinfer_dir = workspace_root.join("openinfer");
    let shaders_json = openinfer_dir.join("src/ops/vulkan/shaders.json");
    let contents =
        fs::read_to_string(&shaders_json).with_context(|| format!("read {}", shaders_json.display()))?;
    let value: Value = serde_json::from_str(&contents)?;
    let ops = value
        .get("ops")
        .and_then(|ops| ops.as_object())
        .ok_or_else(|| anyhow!("shaders.json missing ops object"))?;

    let mut deleted = 0usize;
    for (op, config) in ops {
        let spv_dir = config
            .get("spv_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{} missing spv_dir", op))?;
        let spv_dir = openinfer_dir.join(spv_dir);
        if !spv_dir.exists() {
            continue;
        }
        for entry in fs::read_dir(&spv_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("spv") {
                fs::remove_file(&path)
                    .with_context(|| format!("remove {}", path.display()))?;
                deleted += 1;
            }
        }
    }

    println!("Deleted {} SPIR-V files.", deleted);
    Ok(())
}
