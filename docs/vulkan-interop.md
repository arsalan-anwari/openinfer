Vulkan Interop Overview
=======================

This document describes how the Vulkan backend is wired and how to add custom ops.
The Vulkan runtime uses `ash`, precompiles Slang to SPIR-V at build time, and
embeds the SPIR-V blobs with `include_bytes!` for zero runtime shader compilation.
All Vulkan code, dependencies, and shader compilation are gated behind the
`vulkan` feature.

High-Level Flow
---------------
1) `build.rs` reads `openinfer/src/ops/vulkan/shaders.json`, scans each Slang
   file for compute entry points, compiles each target via `slangc` into a `.spv`
   file, then generates a small Rust module in `OUT_DIR` that embeds those blobs.
2) The Vulkan shader registry loads `shaders.json` and uses the generated
   module to fill `OpShaderInfo` with embedded SPIR-V bytes.
3) Vulkan ops use `VulkanRuntime` to create buffers, pipelines, and dispatch
   compute workloads.

Key Files
---------
- `openinfer/build.rs`
  - Offline Slang compilation for Vulkan shaders (only with `--features vulkan`).
  - Uses `SLANGC` env var or `slangc` in PATH.
- `openinfer/src/backend/vulkan/runtime.rs`
  - `ash` setup, buffer allocation, pipeline creation, dispatch.
- `openinfer/src/backend/vulkan/mod.rs`
  - Shader registry loads `shaders.json` and embeds SPIR-V per target.
- `openinfer/src/ops/vulkan/*`
  - Per-op kernel launchers and Slang shaders.
- `openinfer/src/ops/vulkan/shaders.json`
  - Manifest of shader sources, SPIR-V outputs, push constants, and settings.

Shader Manifest Format
----------------------
`openinfer/src/ops/vulkan/shaders.json` describes each op:

- `path`: path to the Slang shader source (`.slang`).
- `spv_dir`: directory where SPIR-V outputs live for this op.
- `push_constants_size`: byte size of push constants (currently 16).
- `settings`: arbitrary key/value settings passed to kernels.

Example entry:
```json
{
  "ops": {
    "add": {
      "path": "src/ops/vulkan/add/add.slang",
      "spv_dir": "src/ops/vulkan/add/bin",
      "push_constants_size": 16
    }
  }
}
```

Slang Shader Conventions
------------------------
- Each op has a single `.slang` file under `openinfer/src/ops/vulkan/<op>/`.
- SPIR-V output lives under `openinfer/src/ops/vulkan/<op>/bin/`.
- Entry points are compiled per target name (e.g., `add_f32`), but the Vulkan
  runtime uses `main` for pipeline creation while selecting the SPIR-V blob by
  target.
- Push constants:
  - Layout: `uint len`, `uint flags`, `uint pad0`, `uint pad1`
  - Size: 16 bytes

Adding a New Vulkan Op
----------------------
1) Create the op folder and Slang shader:
   - `openinfer/src/ops/vulkan/<op>/mod.rs`
   - `openinfer/src/ops/vulkan/<op>/registry.rs`
   - `openinfer/src/ops/vulkan/<op>/<op>.slang`
2) Ensure the Slang shader defines the compute entry points (e.g. `add_f32`).
   `build.rs` will compile each `[shader("compute")]` entry into:
   - `src/ops/vulkan/<op>/bin/<entry>.spv`
3) Add target selection patterns:
   - `openinfer/src/ops/vulkan/mod.rs` for per-op dispatch.
   - `openinfer/src/ops/vulkan/<op>/mod.rs` for the op-specific matcher.
4) Add the op kernel and registry entries:
   - `openinfer/src/ops/vulkan/<op>/mod.rs` should call `runtime.dispatch(...)`.
   - `openinfer/src/ops/vulkan/<op>/registry.rs` should register the Vulkan kernel.
5) Ensure dtype support is enforced:
   - `openinfer/src/backend/vulkan/mod.rs` has `ensure_supported_dtype(...)`.
   - Return a clean error if the dtype is not supported on the current GPU.

Common Kernel Launcher Pattern
-------------------------------
- Resolve runtime from input buffers.
- Validate shapes/dtypes using settings from `OpShaderInfo`.
- Resolve a SPIR-V target with `spv_target_name(op, dtype, attrs)`.
- Get SPIR-V bytes via `VulkanBuffer::spv_bytes_for_target`.
- Allocate output buffer and dispatch:
  - `runtime.dispatch(op, dtype, target, "main", spirv, input0, input1, output, flags, len)`

Target Naming
-------------
- Default convention is `<op>_<dtype>` (e.g. `add_f32`).
- Each op owns a target matcher (e.g. `spv_target_name_add`) so you can add
  tiered matches per op/dtype/attrs.
- If you add custom attributes, add a match in the op-specific function that
  appends a suffix (e.g. `relu_f32_leaky`).

Notes and Limitations
---------------------
- DType support is limited to the set allowed in `openinfer/src/backend/vulkan/mod.rs`.
  Unsupported dtypes return an error before kernel dispatch.
- `abs` for unsigned/bool can be short-circuited in Rust without launching
  a kernel (see `openinfer/src/ops/vulkan/abs/mod.rs`).
- If you add a dtype or op, update:
  - `shaders.json` (path + spv_dir)
  - op registry + kernel launcher
  - target matcher in the op module
  - dtype checks

Build and Run
-------------
- Build with Vulkan enabled:
  - `cargo build -p openinfer --features vulkan`
- Run the minimal example:
  - `cargo run --example minimal --features vulkan`

Feature Gating
--------------
- Vulkan support is only available when built with `--features vulkan`.
- Without the feature:
  - `ash` is not linked.
  - Vulkan modules and adapters are not compiled.
  - `Device::Vulkan` is rejected as unsupported.
