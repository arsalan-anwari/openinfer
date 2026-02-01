use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::thread;
use std::time::Duration;

use serde_json::Value;

struct OpSpecInfo {
    name: String,
    inplace: bool,
    broadcast: bool,
    accumulate: bool,
}

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let openinfer_dir = workspace_root.join("openinfer");
    let ops_json_path = workspace_root.join("ops.json");
    let specs = match parse_opspecs(&ops_json_path) {
        Ok(specs) => specs,
        Err(err) => {
            eprintln!("build_opspec: failed to parse op defs: {err}");
            std::process::exit(1);
        }
    };
    let total = specs.len();
    let term_width = env::var("COLUMNS")
        .ok()
        .and_then(|value: String| value.parse::<usize>().ok())
        .unwrap_or(120)
        .max(40);
    for (idx, spec) in specs.iter().enumerate() {
        let line = format!(
            "[{}/{}] -- generating opspec for {}[inplace={}, broadcast={}, accumulate={}]",
            idx + 1,
            total,
            spec.name,
            spec.inplace,
            spec.broadcast,
            spec.accumulate
        );
        let line = truncate_to_width(&line, term_width.saturating_sub(1));
        print!("\r{line:<width$}", width = term_width);
        let _ = io::stdout().flush();
        thread::sleep(Duration::from_millis(200));
    }
    println!();
    if let Err(err) = openinfer_generator::op_schema::generate_cpu_kernels(&openinfer_dir) {
        eprintln!("build_opspec: failed to generate cpu kernels: {err}");
        std::process::exit(1);
    }
}

fn parse_opspecs(path: &Path) -> Result<Vec<OpSpecInfo>, String> {
    let contents = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let root: Value = serde_json::from_str(&contents).map_err(|err| err.to_string())?;
    let ops = root
        .get("ops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "ops.json missing ops array".to_string())?;
    let mut specs = Vec::new();
    for op in ops {
        let obj = op
            .as_object()
            .ok_or_else(|| "ops.json op must be object".to_string())?;
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "op missing name".to_string())?
            .to_string();
        let inplace = parse_allow(obj.get("inplace")).map_err(|err| err.to_string())?;
        let broadcast = parse_allow(obj.get("broadcast")).map_err(|err| err.to_string())?;
        let accumulate = parse_allow(obj.get("accumulate")).map_err(|err| err.to_string())?;
        specs.push(OpSpecInfo {
            name,
            inplace,
            broadcast,
            accumulate,
        });
    }
    Ok(specs)
}

fn parse_allow(value: Option<&Value>) -> Result<bool, &'static str> {
    match value.and_then(|v| v.as_str()) {
        Some("allow") => Ok(true),
        Some("deny") => Ok(false),
        _ => Err("unsupported allow/deny value"),
    }
}

fn truncate_to_width(text: &str, width: usize) -> String {
    if text.len() <= width {
        return text.to_string();
    }
    if width <= 3 {
        return text.chars().take(width).collect();
    }
    let keep = width - 3;
    let mut out = text.chars().take(keep).collect::<String>();
    out.push_str("...");
    out
}
