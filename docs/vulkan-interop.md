Vulkan Interop Overview
=======================

This document describes how the Vulkan backend is wired and how to add custom ops.
The Vulkan runtime uses `ash`, precompiles Slang to SPIR-V at build time, and
embeds the SPIR-V blobs with `include_bytes!` for zero runtime shader compilation.
All Vulkan code, dependencies, and shader compilation are gated behind the
`vulkan` feature.

High-Level Flow
---------------
1) `build.rs` reads `openinfer/src/ops/vulkan/shaders.json`. If an op lists
   `shader_files`, only those `.slang` files are compiled; otherwise `build.rs`
   scans all `.slang` files under each op’s `shader_dir`. Each compute entry
   point is compiled via `slangc` into a `.spv` file, then a small Rust module
   is generated in `OUT_DIR`
   that embeds those blobs. Broadcast is a special-case shader with a fixed path
   and SPV output directory (hardcoded in `build.rs`), so it is not listed in
   `shaders.json`.
2) The Vulkan shader registry loads `shaders.json` and uses the generated
   module to fill `OpShaderInfo` with embedded SPIR-V bytes for each op.
3) Vulkan ops use `VulkanRuntime` to create buffers, pipelines, and dispatch
   compute workloads. Pre-processing like broadcast lives in the backend.

Maintainability Notes
---------------------
- `settings.json` at repo root controls `openinfer.vulkan.max_tensor_rank`, and
  `openinfer/build.rs` emits `OPENINFER_VK_MAX_DIMS` plus a generated shader include
  at `openinfer/src/ops/vulkan/shaders/generated_config.slang`.
- `TensorDesc` remains universal across ops, but push constants and descriptor
  bindings are now per-op. Add declares `AddPush` in
  `openinfer/src/ops/vulkan/add/shaders/common.slang`.
- Kernel launchers use a generic `VulkanOpSpec` and `dispatch_compute` entrypoint
  to avoid per-op Vulkan boilerplate.
- The runtime is split across `openinfer/src/ops/vulkan/runtime/*` for device setup,
  buffer management, pipeline creation, and dispatch.

Key Files
---------
- `openinfer/build.rs`
  - Offline Slang compilation for Vulkan shaders (only with `--features vulkan`).
  - Uses `SLANGC` env var or `slangc` in PATH.
- `openinfer/src/backend/vulkan/runtime.rs`
  - `ash` setup, buffer allocation, pipeline creation, dispatch.
- `openinfer/src/backend/vulkan/mod.rs`
  - Shader registry loads `shaders.json` and embeds SPIR-V per target.
- `openinfer/src/backend/vulkan/broadcast.rs`
  - Broadcast preprocessing for Vulkan tensors.
- `openinfer/src/ops/vulkan/*`
  - Per-op kernel launchers and Slang shaders.
- `openinfer/src/ops/vulkan/shaders.json`
  - Manifest of shader sources, SPIR-V outputs, push constants, and settings.

Shader Manifest Format
----------------------
`openinfer/src/ops/vulkan/shaders.json` describes each op (broadcast is omitted
because its shader path/output are hardcoded in `build.rs`):

- `shader_dir`: directory containing shader files for the op.
- `shader_files` (optional): list of `.slang` files (relative to `shader_dir`)
  to compile. When present, this overrides recursive scanning.
- `spv_dir`: directory where SPIR-V outputs live for this op.
- `push_constants_size`: byte size of push constants (currently 16).
- `settings`: arbitrary key/value settings passed to kernels.

Example entry:
```json
{
  "ops": {
    "add": {
      "shader_dir": "src/ops/vulkan/add/shaders",
      "spv_dir": "src/ops/vulkan/add/bin",
      "push_constants_size": 16
    }
  }
}
```

Slang Shader Conventions
------------------------
- Each op has shaders under:
  - `openinfer/src/ops/vulkan/<op>/shaders/base/`
  - `openinfer/src/ops/vulkan/<op>/shaders/inplace/`
  - `openinfer/src/ops/vulkan/<op>/shaders/accumulate/`
- Dtypes are consolidated into shared Slang files (e.g. `add_signed.slang`,
  `add_float.slang`, `add_signed_packed.slang`) with multiple entry points
  per file instead of one file per dtype.
- SPIR-V output lives under `openinfer/src/ops/vulkan/<op>/bin/`.
- Broadcast is stored under `openinfer/src/backend/vulkan/broadcast/` with SPIR-V
  emitted to `backend/vulkan/broadcast/bin/`.
- Entry points are compiled per target name (e.g., `add_f32`), but the Vulkan
  runtime uses `main` for pipeline creation while selecting the SPIR-V blob by
  target.
- Packed dtypes use `ByteAddressBuffer` with shared helpers in
  `openinfer/src/ops/vulkan/packed_utils.slang`.
- Low-bit float helpers live in `openinfer/src/ops/vulkan/float_utils.slang`.
- Push constants:
  - Layout: `uint len`, `uint flags`, `uint pad0`, `uint pad1`
  - Size: 16 bytes
- Descriptor bindings:
  - Binding 0: input0
  - Binding 1: input1 (for unary ops, runtime binds output here as well)
  - Binding 2: output

Adding a New Vulkan Op
----------------------
1) Create the op folder and Slang shader(s):
   - `openinfer/src/ops/vulkan/<op>/mod.rs`
   - `openinfer/src/ops/vulkan/<op>/registry.rs`
   - Consolidated shaders in `shaders/{base,inplace,accumulate}/` (multiple
     entry points per file), or per-dtype shaders if you prefer.
2) Ensure each `.slang` file defines a compute entry point that matches the
   entry point name (e.g. `add_i8` in `add_signed.slang`).
   `build.rs` compiles each `[shader("compute")]` entry into:
   - `src/ops/vulkan/<op>/bin/<entry>.spv`
3) Add the op to `openinfer/src/ops/vulkan/shaders.json` so it is compiled and
   embedded (broadcast is the only built-in exception).
4) Add target selection patterns:
   - `openinfer/src/ops/vulkan/mod.rs` for per-op dispatch.
   - `openinfer/src/ops/vulkan/<op>/mod.rs` for the op-specific matcher.
5) Add the op kernel and registry entries:
   - `openinfer/src/ops/vulkan/<op>/mod.rs` should call `runtime.dispatch(...)`.
   - `openinfer/src/ops/vulkan/<op>/registry.rs` should register the Vulkan kernel.
6) Ensure dtype support is enforced:
   - `openinfer/src/backend/vulkan/mod.rs` checks device feature flags and dtype rules.
   - Return a clean error if the dtype is not supported on the current GPU.

Common Kernel Launcher Pattern
-------------------------------
- Resolve runtime from input buffers.
- Validate shapes/dtypes using settings from `OpShaderInfo`.
- Resolve a SPIR-V target with `spv_target_name(op, dtype, attrs)`.
- Get SPIR-V bytes via `VulkanBuffer::spv_bytes_for_target`.
- Allocate output buffer and dispatch:
  - `runtime.dispatch(op, dtype, target, "main", spirv, input0, input1, output, flags, len)`
  - The runtime derives the descriptor set index from the SPIR-V and binds the
    descriptor set at that index.

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
- i64/u64 require `shader_int64` support from the Vulkan device.
- f64 requires `shader_float64` support from the Vulkan device.
- f16 uses native half when `shader_float16` is available; otherwise shaders cast
  to f32 and write back. Use `Simulator::with_simulated_float()` to force the
  simulated f16 path even when native support is available.
- f8/bf16 are always cast to f32 in shaders (no native support assumed).
- Packed integer types (i1/i2/i4/u1/u2/u4) are stored as packed bits in buffers;
  Vulkan shaders decode/operate/encode in-place using byte-addressed buffers.
- `t1`/`t2` are not supported in Vulkan.
- The simulator uses in-place kernels automatically when output aliases an
  input and the op supports it (e.g. `op add(x, w) >> x`). On Vulkan this can
  be noticeably faster because it avoids extra allocations and synchronization.
- If the device lacks `shader_int64` or `shader_float64`, Vulkan kernels fall
  back to CPU with a warning instead of erroring.
- If you add a dtype or op, update:
  - `shaders.json` (path + spv_dir)
  - op registry + kernel launcher
  - target matcher in the op module
  - dtype checks

Build and Run
-------------
- Build with Vulkan enabled:
  - `cargo build -p openinfer --features vulkan`
- Build with Vulkan shader progress output:
  - `cargo build-spv`
- Run the minimal example:
  - `cargo run --example minimal --features vulkan`

Feature Gating
--------------
- Vulkan support is only available when built with `--features vulkan`.
- Without the feature:
  - `ash` is not linked.
  - Vulkan modules and adapters are not compiled.
  - `Device::Vulkan` is rejected as unsupported.
