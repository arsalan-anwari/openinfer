use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

fn main() -> Result<()> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let openinfer_dir = workspace_root.join("openinfer");
    let ops_json = workspace_root.join("ops.json");
    let contents =
        fs::read_to_string(&ops_json).with_context(|| format!("read {}", ops_json.display()))?;
    let value: Value = serde_json::from_str(&contents)?;
    let ops = value
        .get("ops")
        .and_then(|ops| ops.as_array())
        .ok_or_else(|| anyhow!("ops.json missing ops array"))?;

    let mut deleted = 0usize;
    for op in ops {
        let op_name = op
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("ops.json op missing name"))?;
        let vulkan = match op
            .get("devices")
            .and_then(|v| v.as_object())
            .and_then(|v| v.get("vulkan"))
            .and_then(|v| v.as_object())
        {
            Some(vulkan) => vulkan,
            None => continue,
        };
        let spv_dir = vulkan
            .get("spv_dir")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("{op_name} missing spv_dir"))?;
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
