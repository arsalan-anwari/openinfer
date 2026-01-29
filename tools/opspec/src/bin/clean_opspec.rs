use std::fs;
use std::path::Path;

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("missing workspace root");
    let cpu_ops_dir = workspace_root.join("openinfer/src/ops/cpu");
    let entries = match fs::read_dir(&cpu_ops_dir) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!(
                "clean_opspec: failed to read cpu ops dir {}: {err}",
                cpu_ops_dir.display()
            );
            std::process::exit(1);
        }
    };
    let mut cleaned = 0usize;
    for entry in entries.flatten() {
        let path = entry.path().join("kernel.rs");
        if path.exists() {
            let metadata = match fs::metadata(&path) {
                Ok(meta) => meta,
                Err(err) => {
                    eprintln!("clean_opspec: failed to stat {}: {err}", path.display());
                    std::process::exit(1);
                }
            };
            if metadata.len() > 0 {
                if let Err(err) = fs::write(&path, "") {
                    eprintln!("clean_opspec: failed to clear {}: {err}", path.display());
                    std::process::exit(1);
                }
                cleaned += 1;
            }
        }
    }
    if cleaned == 0 {
        println!("clean-opspec: no opspec to clean");
    } else {
        println!("clean-opspec: cleared {cleaned} kernel.rs files");
    }
}
