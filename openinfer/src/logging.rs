use std::env;
use std::fmt::Arguments;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
enum TraceLevel {
    Off,
    Basic,
    Full,
}

const COLOR_WARNING: &str = "33";
const COLOR_ERROR: &str = "91";
const COLOR_CRITICAL: &str = "31";
const COLOR_VK_TRACE: &str = "32";
const COLOR_TRACE: &str = "34";

static TRACE_LEVEL: OnceLock<TraceLevel> = OnceLock::new();
static VULKAN_TRACE_LEVEL: OnceLock<TraceLevel> = OnceLock::new();

fn parse_trace_level(value: &str) -> TraceLevel {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" => TraceLevel::Basic,
        "full" => TraceLevel::Full,
        _ => TraceLevel::Off,
    }
}

fn env_trace_level(name: &str, cache: &OnceLock<TraceLevel>) -> TraceLevel {
    *cache.get_or_init(|| {
        env::var(name)
            .ok()
            .as_deref()
            .map(parse_trace_level)
            .unwrap_or(TraceLevel::Off)
    })
}

fn trace_level() -> TraceLevel {
    env_trace_level("OPENINFER_TRACE", &TRACE_LEVEL)
}

fn vulkan_trace_level() -> TraceLevel {
    env_trace_level("OPENINFER_VULKAN_TRACE", &VULKAN_TRACE_LEVEL)
}

fn trace_full_enabled() -> bool {
    matches!(trace_level(), TraceLevel::Full)
}

fn trace_basic_enabled() -> bool {
    matches!(trace_level(), TraceLevel::Full | TraceLevel::Basic)
}

fn vk_trace_enabled() -> bool {
    matches!(vulkan_trace_level(), TraceLevel::Full | TraceLevel::Basic)
}

fn timestamp_hms() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        % 86_400;
    let hours = secs / 3_600;
    let minutes = (secs % 3_600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

fn emit(kind: &str, color: &str, args: Arguments) {
    let ts = timestamp_hms();
    if color.is_empty() {
        println!("{ts} [{kind}] -- {args}");
    } else {
        println!("{ts} [\u{001b}[{color}m{kind}\u{001b}[0m] -- {args}");
    }
}

/// Emit a warning message when trace level allows it.
pub fn emit_warning(args: Arguments) {
    if trace_full_enabled() {
        emit("WARNING", COLOR_WARNING, args);
    }
}

/// Emit an error message when trace level allows it.
pub fn emit_error(args: Arguments) {
    if trace_basic_enabled() {
        emit("ERROR", COLOR_ERROR, args);
    }
}

/// Emit a Vulkan trace message when Vulkan trace is enabled.
pub fn emit_vk_trace(args: Arguments) {
    if vk_trace_enabled() {
        emit("VK_TRACE", COLOR_VK_TRACE, args);
    }
}

/// Emit a critical message unconditionally.
pub fn emit_critical(args: Arguments) {
    emit("CRITICAL", COLOR_CRITICAL, args);
}

/// Emit a trace message when trace level allows it.
pub fn emit_trace(args: Arguments) {
    if trace_basic_enabled() {
        emit("TRACE", COLOR_TRACE, args);
    }
}

/// Emit a warning message via the logging subsystem.
#[macro_export]
macro_rules! warning {
    ($($arg:tt)*) => {
        $crate::logging::emit_warning(format_args!($($arg)*))
    };
}

/// Emit an error message via the logging subsystem.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::logging::emit_error(format_args!($($arg)*))
    };
}

/// Emit a Vulkan trace message via the logging subsystem.
#[macro_export]
macro_rules! vk_trace {
    ($($arg:tt)*) => {
        $crate::logging::emit_vk_trace(format_args!($($arg)*))
    };
}

/// Emit a critical message via the logging subsystem.
#[macro_export]
macro_rules! critical {
    ($($arg:tt)*) => {
        $crate::logging::emit_critical(format_args!($($arg)*))
    };
}

/// Emit a trace message via the logging subsystem.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        $crate::logging::emit_trace(format_args!($($arg)*))
    };
}

/// Emit a raw log message using `println!`.
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        println!($($arg)*)
    };
}
