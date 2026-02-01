use std::fs;
use std::path::Path;

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let cpu_ops_dirs = [
        workspace_root.join("openinfer/src/ops/cpu"),
        workspace_root.join("openinfer/ops/cpu"),
    ];
    let mut cleaned = 0usize;
    for cpu_ops_dir in cpu_ops_dirs {
        if !cpu_ops_dir.exists() {
            continue;
        }
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
        for entry in entries.flatten() {
            let category_path = entry.path();
            if !category_path.is_dir() {
                continue;
            }
            let ops = match fs::read_dir(&category_path) {
                Ok(ops) => ops,
                Err(err) => {
                    eprintln!(
                        "clean_opspec: failed to read cpu category dir {}: {err}",
                        category_path.display()
                    );
                    std::process::exit(1);
                }
            };
            for op_entry in ops.flatten() {
                let op_path = op_entry.path();
                if !op_path.is_dir() {
                    continue;
                }
                let path = op_path.join("kernel.rs");
                if path.ends_with("src/ops/cpu/casting/cast/kernel.rs")
                    || path.ends_with("ops/cpu/casting/cast/kernel.rs")
                {
                    continue;
                }
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
        }
    }
    if cleaned == 0 {
        println!("clean-opspec: no opspec to clean");
    } else {
        println!("clean-opspec: cleared {cleaned} kernel.rs files");
    }
}
