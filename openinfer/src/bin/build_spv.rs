use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct VulkanShaderManifest {
    ops: std::collections::HashMap<String, VulkanShaderEntry>,
}

#[derive(Debug, Deserialize)]
struct VulkanShaderEntry {
    shader_dir: Option<String>,
    spv_dir: String,
}

const BROADCAST_OP: &str = "broadcast";
const BROADCAST_PATH: &str = "src/backend/vulkan/broadcast/broadcast.slang";
const BROADCAST_SPV_DIR: &str = "src/backend/vulkan/broadcast/bin";

struct ProgressLine {
    last_len: usize,
}

impl ProgressLine {
    fn new() -> Self {
        Self { last_len: 0 }
    }

    fn print(&mut self, message: &str) -> io::Result<()> {
        let mut stderr = io::stderr();
        let pad = self.last_len.saturating_sub(message.len());
        write!(stderr, "\r{}{}", message, " ".repeat(pad))?;
        stderr.flush()?;
        self.last_len = message.len();
        Ok(())
    }
}

impl Drop for ProgressLine {
    fn drop(&mut self) {
        if self.last_len > 0 {
            let _ = writeln!(io::stderr());
            self.last_len = 0;
        }
    }
}

struct CompileJob {
    op_name: String,
    src_path: PathBuf,
    spv_path: PathBuf,
    target: String,
}

fn main() -> Result<()> {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
    let manifest_path = manifest_dir.join("src/ops/vulkan/shaders.json");
    let contents = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: VulkanShaderManifest = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;

    let include_broadcast = !manifest.ops.contains_key(BROADCAST_OP);
    let mut jobs = Vec::new();
    for (op_name, entry) in &manifest.ops {
        jobs.extend(gather_jobs_for_op(&manifest_dir, op_name, entry)?);
    }
    if include_broadcast {
        let broadcast_entry = VulkanShaderEntry {
            shader_dir: None,
            spv_dir: BROADCAST_SPV_DIR.to_string(),
        };
        jobs.extend(gather_jobs_for_op(
            &manifest_dir,
            BROADCAST_OP,
            &broadcast_entry,
        )?);
    }

    let total = jobs.len();
    let mut progress = ProgressLine::new();
    let max_width = terminal_width().unwrap_or(120);
    for (idx, job) in jobs.iter().enumerate() {
        let spv_display = display_path(&job.spv_path, &manifest_dir);
        let label = format!(
            "[{}/{}] {}:{} -> {}",
            idx + 1,
            total,
            job.op_name,
            job.target,
            spv_display
        );
        let label = truncate_line(&label, max_width);
        progress
            .print(&label)
            .with_context(|| format!("failed to print progress for {}", job.spv_path.display()))?;
        compile_slang(&job.src_path, &job.spv_path, &job.target)
            .with_context(|| format!("failed to compile {}", job.op_name))?;
    }

    let status = run_cargo_build()?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

fn run_cargo_build() -> Result<ExitStatus> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build");
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut quiet_build = true;
    if let Some(pos) = args.iter().position(|arg| arg == "--no-quiet-build") {
        args.remove(pos);
        quiet_build = false;
    }
    if let Some(pos) = args.iter().position(|arg| arg == "--quiet-build") {
        args.remove(pos);
        quiet_build = true;
    }

    if quiet_build {
        cmd.arg("--quiet");
    }

    if args.is_empty() {
        cmd.arg("-p").arg("openinfer");
        cmd.arg("--features").arg("vulkan");
    } else {
        for arg in args {
            cmd.arg(arg);
        }
    }
    cmd.status().with_context(|| "failed to run cargo build")
}

fn gather_jobs_for_op(
    manifest_dir: &Path,
    op_name: &str,
    entry: &VulkanShaderEntry,
) -> Result<Vec<CompileJob>> {
    let mut jobs = Vec::new();
    let paths = shader_paths(manifest_dir, entry, op_name)?;
    let spv_dir = manifest_dir.join(&entry.spv_dir);
    for src_path in paths {
        let mut includes = Vec::new();
        collect_includes(&src_path, &mut includes, &mut HashSet::new())
            .with_context(|| format!("failed to parse includes for {}", src_path.display()))?;
        let targets = slang_entry_points(&src_path)
            .with_context(|| format!("failed to parse entry points for {}", src_path.display()))?;
        for target in &targets {
            let spv_path = spv_dir.join(format!("{}.spv", target));
            let should_compile = needs_rebuild(&src_path, &spv_path, &includes)?;
            if should_compile {
                jobs.push(CompileJob {
                    op_name: op_name.to_string(),
                    src_path: src_path.clone(),
                    spv_path,
                    target: target.clone(),
                });
            }
        }
    }
    Ok(jobs)
}

fn needs_rebuild(src: &Path, spv: &Path, includes: &[PathBuf]) -> Result<bool> {
    if !spv.exists() {
        return Ok(true);
    }
    let src_meta = fs::metadata(src).with_context(|| format!("failed to stat {}", src.display()))?;
    let spv_meta = fs::metadata(spv).with_context(|| format!("failed to stat {}", spv.display()))?;
    let src_time = src_meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let spv_time = spv_meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    if src_time > spv_time {
        return Ok(true);
    }
    for include in includes {
        let meta =
            fs::metadata(include).with_context(|| format!("failed to stat {}", include.display()))?;
        let inc_time = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if inc_time > spv_time {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_includes(src: &Path, includes: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>) -> Result<()> {
    let contents = fs::read_to_string(src).with_context(|| format!("failed to read {}", src.display()))?;
    let parent = src
        .parent()
        .ok_or_else(|| anyhow!("missing parent for {}", src.display()))?;
    for line in contents.lines() {
        let trimmed = line.trim();
        let rest = match trimmed.strip_prefix("#include") {
            Some(rest) => rest.trim(),
            None => continue,
        };
        let start = match rest.find('"') {
            Some(idx) => idx + 1,
            None => continue,
        };
        let end = match rest[start..].find('"') {
            Some(idx) => start + idx,
            None => continue,
        };
        let include_path = &rest[start..end];
        let mut path = PathBuf::from(include_path);
        if !path.is_absolute() {
            path = parent.join(path);
        }
        let path = fs::canonicalize(&path)
            .with_context(|| format!("failed to resolve include {}", path.display()))?;
        if seen.insert(path.clone()) {
            includes.push(path.clone());
            collect_includes(&path, includes, seen)?;
        }
    }
    Ok(())
}

fn compile_slang(src: &Path, spv: &Path, entry_point: &str) -> Result<()> {
    if let Some(parent) = spv.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_string());
    let status = Command::new(&slangc)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("sm_6_2")
        .arg("-stage")
        .arg("compute")
        .arg("-entry")
        .arg(entry_point)
        .arg("-o")
        .arg(spv)
        .arg(src)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| {
            format!(
                "failed to spawn slangc for {} (set SLANGC or add slangc to PATH)",
                src.display()
            )
        })?;

    if !status.success() {
        return Err(anyhow!("slangc failed for {}", src.display()));
    }

    Ok(())
}

fn slang_entry_points(src: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(src).with_context(|| format!("failed to read {}", src.display()))?;
    let mut targets = Vec::new();
    let mut awaiting_entry = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[shader(\"compute\")]") {
            awaiting_entry = true;
            continue;
        }
        if !awaiting_entry {
            continue;
        }
        if trimmed.starts_with('[') || trimmed.is_empty() {
            continue;
        }
        if let Some(idx) = trimmed.find("void ") {
            let after = &trimmed[idx + 5..];
            let name = after
                .split(|c: char| c == '(' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                targets.push(name.to_string());
                awaiting_entry = false;
            }
        }
    }
    if targets.is_empty() {
        return Err(anyhow!("no compute entry points found in {}", src.display()));
    }
    targets.sort();
    targets.dedup();
    Ok(targets)
}

fn shader_paths(
    manifest_dir: &Path,
    entry: &VulkanShaderEntry,
    op_name: &str,
) -> Result<Vec<PathBuf>> {
    if op_name == BROADCAST_OP {
        return Ok(vec![manifest_dir.join(BROADCAST_PATH)]);
    }
    let shader_dir = entry
        .shader_dir
        .as_ref()
        .ok_or_else(|| anyhow!("missing shader_dir for op {}", op_name))?;
    let shader_dir = manifest_dir.join(shader_dir);
    if !shader_dir.exists() {
        return Err(anyhow!(
            "shader_dir does not exist for op {}: {}",
            op_name,
            shader_dir.display()
        ));
    }
    let mut paths = Vec::new();
    collect_slang_files(&shader_dir, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_slang_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_slang_files(&path, out)?;
        } else if path.extension().map(|ext| ext == "slang").unwrap_or(false) {
            out.push(path);
        }
    }
    Ok(())
}

fn display_path(path: &Path, base: &Path) -> String {
    if let Ok(stripped) = path.strip_prefix(base) {
        return stripped.display().to_string();
    }
    path.display().to_string()
}

fn terminal_width() -> Option<usize> {
    env::var("COLUMNS").ok().and_then(|value| value.parse::<usize>().ok())
}

fn truncate_line(line: &str, max_width: usize) -> String {
    if max_width == 0 || line.len() <= max_width {
        return line.to_string();
    }
    if max_width <= 3 {
        return ".".repeat(max_width);
    }
    let keep = max_width - 3;
    format!("{}...", &line[..keep])
}
