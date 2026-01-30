use openinfer::{graph, GraphSerialize};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            a: f32[B];
            y: f32[B] @init(5.0);
        }

        block entry {
            assign t0: f32[B];
            op add(x, a) >> t0;
            op mul(y, t0) >> y;
            return;
        }
    };

    let json_value = GraphSerialize::json(&g)?;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/openinfer/out");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("minimal-graph.json");
    std::fs::write(out_path, serde_json::to_string_pretty(&json_value)?)?;

    Ok(())
}
