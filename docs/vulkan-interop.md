Vulkan Interop Overview
=======================

This document describes how the Vulkan backend is wired and how to add custom ops.
The Vulkan runtime uses `ash`, precompiles Slang to SPIR-V at build time, and
embeds the SPIR-V blobs with `include_bytes!` for zero runtime shader compilation.

High-Level Flow
---------------
1) `build.rs` reads `openinfer/src/ops/vulkan/shaders.json` and compiles each
   op+dtype entry via `slangc` into a `.spv` file.
2) The Vulkan shader registry loads `shaders.json` and embeds SPIR-V bytes into
   `OpShaderInfo` via `include_bytes!`.
3) Vulkan ops use `VulkanRuntime` to create buffers, pipelines, and dispatch
   compute workloads.

Key Files
---------
- `openinfer/build.rs`
  - Offline Slang compilation for Vulkan when `--features vulkan` is enabled.
  - Uses `SLANGC` env var or `slangc` in PATH.
- `openinfer/src/backend/vulkan/runtime.rs`
  - `ash` setup, buffer allocation, pipeline creation, dispatch.
- `openinfer/src/backend/vulkan/mod.rs`
  - Shader registry loads `shaders.json` and embeds SPIR-V per dtype.
- `openinfer/src/ops/vulkan/*`
  - Per-op kernel launchers and Slang shaders.
- `openinfer/src/ops/vulkan/shaders.json`
  - Manifest of shader sources, SPIR-V outputs, push constants, and settings.

Shader Manifest Format
----------------------
`openinfer/src/ops/vulkan/shaders.json` describes each op:

- `path`: path to the Slang shader source (`.slang`).
- `spv_by_dtype`: map of dtype identifier to output SPIR-V path.
- `push_constants_size`: byte size of push constants (currently 16).
- `settings`: arbitrary key/value settings passed to kernels.

Example entry:
```json
{
  "ops": {
    "add": {
      "path": "src/ops/vulkan/add/add.slang",
      "spv_by_dtype": {
        "f32": "src/ops/vulkan/add/bin/add_f32.spv",
        "i32": "src/ops/vulkan/add/bin/add_i32.spv"
      },
      "push_constants_size": 16,
      "settings": {
        "strict_shapes": true,
        "allow_mixed_dtypes": false
      }
    }
  }
}
```

Slang Shader Conventions
------------------------
- Each op has a single `.slang` file under `openinfer/src/ops/vulkan/<op>/`.
- SPIR-V output lives under `openinfer/src/ops/vulkan/<op>/bin/`.
- Entry points are compiled per dtype but all use the same exported name `main`.
  The build system sets the Slang entry point name but the compiled SPIR-V
  exports `main`, so the Vulkan runtime uses `main` for pipeline creation.
- Push constants:
  - Layout: `uint len`, `uint flags`, `uint pad0`, `uint pad1`
  - Size: 16 bytes

Adding a New Vulkan Op
----------------------
1) Create the op folder and Slang shader:
   - `openinfer/src/ops/vulkan/<op>/mod.rs`
   - `openinfer/src/ops/vulkan/<op>/registry.rs`
   - `openinfer/src/ops/vulkan/<op>/<op>.slang`
2) Add SPIR-V outputs in `shaders.json` under `spv_by_dtype`, e.g.:
   - `src/ops/vulkan/<op>/bin/<op>_f32.spv`
3) Update the shader registry to embed the SPIR-V:
   - `openinfer/src/backend/vulkan/mod.rs` in `embedded_spirv_for_op`.
4) Add the op kernel and registry entries:
   - `openinfer/src/ops/vulkan/<op>/mod.rs` should call `runtime.dispatch(...)`.
   - `openinfer/src/ops/vulkan/<op>/registry.rs` should register the Vulkan kernel.
5) Ensure dtype support is enforced:
   - `openinfer/src/executor/vulkan.rs` has `ensure_supported_dtype(...)`.
   - Return a clean error if the dtype is not supported on the current GPU.

Common Kernel Launcher Pattern
-------------------------------
- Resolve runtime from input buffers.
- Validate shapes/dtypes using settings from `OpShaderInfo`.
- Get SPIR-V bytes via `VulkanBuffer::spv_bytes_for_dtype`.
- Allocate output buffer and dispatch:
  - `runtime.dispatch(op, dtype, "main", spirv, input0, input1, output, flags, len)`

Notes and Limitations
---------------------
- DType support is limited to the set allowed in `executor/vulkan.rs`.
  Unsupported dtypes return an error before kernel dispatch.
- `abs` for unsigned/bool can be short-circuited in Rust without launching
  a kernel (see `openinfer/src/ops/vulkan/abs/mod.rs`).
- If you add a dtype or op, update:
  - `shaders.json`
  - `embedded_spirv_for_op`
  - op registry + kernel launcher
  - dtype checks

Build and Run
-------------
- Build with Vulkan enabled:
  - `cargo build -p openinfer --features vulkan`
- Run the minimal example:
  - `cargo run --example minimal`
