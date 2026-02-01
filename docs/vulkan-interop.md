Vulkan Interop Overview
=======================

This document describes how the Vulkan backend is wired and how to add custom ops.
The Vulkan runtime uses `ash` and consumes precompiled SPIR-V blobs that are
embedded with `include_bytes!` for zero runtime shader compilation. Vulkan code
and the embedded SPIR-V map are only available with the `vulkan` feature.

High-Level Flow
---------------
1) `cargo build-spv` (from `openinfer-generator`) reads
   `openinfer/src/ops/vulkan/shaders.json`, parses Slang entrypoints, and
   compiles them with `slangc` into per-op `.spv` files under each `spv_dir`.
   Compile-time feature flags come from `settings.json` (or are probed from the
   local Vulkan driver if settings are absent).
2) `openinfer/build.rs` reads `shaders.json` and generates
   `OUT_DIR/spv_embedded.rs`, mapping entrypoint names to the existing `.spv`
   blobs (or an empty map if none are present). It also emits shader/runtime
   config for `OPENINFER_VK_MAX_DIMS`.
3) Vulkan ops use the embedded SPIR-V map to create pipelines and dispatch
   workloads via `openinfer/src/ops/vulkan/runtime/`.

Maintainability Notes
---------------------
- `settings.json` at repo root controls `openinfer.vulkan.max_tensor_rank` and
  optional feature flags (`has_f64`, `has_i64`, `has_u64`). The build
  emits `OPENINFER_VK_MAX_DIMS`, generates
  `openinfer/src/ops/vulkan/shaders/generated_config.slang`, and writes
  `OUT_DIR/vulkan_config.rs`.
- `TensorDesc` remains universal across ops, but push constants and descriptor
  bindings are per-op. For example, add declares `AddPush` in
  `openinfer/src/ops/vulkan/arithmetic/add/shaders/common.slang`.
- Kernel launchers use `VulkanOpSpec` and `dispatch_compute` to avoid per-op
  Vulkan boilerplate.
- The runtime lives under `openinfer/src/ops/vulkan/runtime/` for device setup,
  buffer management, pipeline creation, and dispatch.

Key Files
---------
- `openinfer/build.rs`
  - Emits Vulkan config and embedded SPIR-V map.
- `openinfer/src/ops/vulkan/runtime/`
  - `ash` setup, buffer allocation, pipeline creation, dispatch.
- `openinfer/src/ops/vulkan/spv.rs`
  - Embedded SPIR-V map generated in `OUT_DIR`.
- `openinfer/src/ops/vulkan/*`
  - Per-op kernel launchers and Slang shaders.
- `openinfer/src/ops/vulkan/shaders.json`
  - Manifest of shader sources and SPIR-V outputs.

Shader Manifest Format
----------------------
`openinfer/src/ops/vulkan/shaders.json` describes each op:

- `shader_dir`: directory containing shader files for the op.
- `shader_files`: list of `.slang` files (relative to `shader_dir`) to compile.
- `spv_dir`: directory where SPIR-V outputs live for this op.

Example entry:
```json
{
  "ops": {
    "add": {
      "shader_dir": "src/ops/vulkan/arithmetic/add/shaders",
      "shader_files": ["normal.slang", "accumulate.slang", "packed.slang"],
      "spv_dir": "src/ops/vulkan/arithmetic/add/bin"
    }
  }
}
```

Slang Shader Conventions
------------------------
- Each op has shaders under `openinfer/src/ops/vulkan/<category>/<op>/shaders/`.
  Common patterns use `normal.slang`, `accumulate.slang`, and `packed.slang`.
- Entry points are compiled per target name (e.g., `add_f32`), and the runtime
  selects the appropriate SPIR-V by target name.
- SPIR-V output lives under `openinfer/src/ops/vulkan/<category>/<op>/bin/` after running
  `cargo build-spv`.
- Packed dtypes use `ByteAddressBuffer` with shared helpers in
  `openinfer/src/ops/vulkan/shaders/packed_utils.slang`.
- Low-bit float helpers live in `openinfer/src/ops/vulkan/shaders/float_utils.slang`.
- Push constants are defined per op in `shaders/common.slang` (see `AddPush`).
- Descriptor bindings (current convention):
  - Binding 0: `TensorDesc` array
  - Binding 1: input data buffer (all inputs packed in one buffer)
  - Binding 2: output buffer

Adding a New Vulkan Op
----------------------
1) Create the op folder and Slang shader(s):
   - `openinfer/src/ops/vulkan/<category>/<op>/mod.rs`
   - `openinfer/src/ops/vulkan/<category>/<op>/registry.rs`
   - Shaders under `openinfer/src/ops/vulkan/<category>/<op>/shaders/` (multiple entrypoints
     per file is preferred).
2) Ensure each `.slang` file defines a compute entrypoint that matches the
   target name (e.g. `add_f32_normal`).
3) Add the op to `openinfer/src/ops/vulkan/shaders.json` so `cargo build-spv`
   generates `src/ops/vulkan/<category>/<op>/bin/<entry>.spv`.
4) Add target selection patterns:
   - `openinfer/src/ops/vulkan/mod.rs` for per-op dispatch.
   - `openinfer/src/ops/vulkan/<category>/<op>/mod.rs` for the op-specific matcher.
5) Add the op kernel and registry entries:
   - `openinfer/src/ops/vulkan/<category>/<op>/mod.rs` should call `runtime.dispatch(...)`.
   - `openinfer/src/ops/vulkan/<category>/<op>/registry.rs` should register the Vulkan kernel.
6) Ensure dtype support is enforced:
   - `openinfer/src/ops/vulkan/mod.rs` checks feature flags and dtype rules.
   - Return a clean error if the dtype is not supported on the current GPU.

Common Kernel Launcher Pattern
-------------------------------
- Resolve `VulkanRuntime` from input buffers.
- Validate shapes/dtypes and select an entrypoint name.
- Build a `VulkanOpSpec` with the entry name, `spv_dir`, workgroup size, and
  push-constant size.
- Prepare bindings (`BindingBytes`) and call `dispatch_compute`, which loads the
  embedded SPIR-V (or falls back to the on-disk `.spv` if not embedded).

Target Naming
-------------
- Default convention is `<op>_<dtype>_normal` (e.g. `add_f32_normal`).
- In-place kernels use `_inplace`, and accumulation kernels use
  `_accumulate_<out_dtype>`.
- Packed dtypes use `_packed` (e.g. `add_i4_packed_inplace`).
- The naming logic lives in `openinfer/src/ops/vulkan/op_helpers.rs` via
  `target_name`.

Notes and Limitations
---------------------
- DType support is limited to the set allowed in `openinfer/src/ops/vulkan/mod.rs`.
  Unsupported dtypes return an error before kernel dispatch.
- i64/u64 require `shader_int64` support from the Vulkan device.
- f64 requires `shader_float64` support from the Vulkan device.
- f16 is always cast to f32 in shaders and written back to f16 (no native half path).
- f8/bf16 are always cast to f32 in shaders (no native support assumed).
- Packed integer types (i1/i2/i4/u1/u2/u4) are stored as packed bits in buffers;
  Vulkan shaders decode/operate/encode in-place using byte-addressed buffers.
- `t1`/`t2` are not supported in Vulkan.
- The simulator uses in-place kernels automatically when output aliases an
  input and the op supports it (e.g. `op add(x, w) >> x`). On Vulkan this can
  be noticeably faster because it avoids extra allocations and synchronization.
- If the device lacks `shader_int64` or `shader_float64`, Vulkan kernels fall
  back to CPU with a warning instead of erroring.
- If tensor rank exceeds `OPENINFER_VK_MAX_DIMS`, Vulkan kernels fall back to CPU.
- If you add a dtype or op, update:
  - `shaders.json` (path + spv_dir)
  - op registry + kernel launcher
  - target matcher in the op module
  - dtype checks

Build and Run
-------------
- Build with Vulkan enabled:
  - `cargo build -p openinfer --features vulkan`
- Build SPIR-V from Slang sources:
  - `cargo build-spv` (requires `slangc` in PATH)
- Run the minimal example:
  - `cargo run --example minimal --features vulkan`

Feature Gating
--------------
- Vulkan support is only available when built with `--features vulkan`.
- Without the feature:
  - `ash` is not linked.
  - Vulkan modules and adapters are not compiled.
  - `Device::Vulkan` is rejected as unsupported.
