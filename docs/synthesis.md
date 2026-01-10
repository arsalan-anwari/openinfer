# Backend Documentation

See backend-specific docs for implementation details and extension points:

- [docs/vulkan-interop.md](vulkan-interop.md)

## Supported Ops

For a current list of supported ops and per-backend dtype coverage, see:

- [docs/ops.md](ops.md)

## Synthesis

Once validated, the same graph can be synthezised to source code with:

* Device-specific scheduling
* Kernel fusion
* Memory planning
* Backend code generation

The output is **plain source code** (C, GLSL shaders, etc.), **not** a binary.

### Passing Graphs to the Synthesizer

The synthesizer accepts either:

* a `Graph` object directly (in-process), or
* a serialized JSON graph (tooling / CLI / build pipelines)

#### In-process usage

```rust
use openinfer::{Synthesizer, Device};

let dev = Device::Vulkan;
let synth = Synthesizer::new(dev);

let plan = synth.synthesize(&model, &graph)?;
plan.emit("build/out")?;
```

### Device Architecture JSON (Synthesizer Input)

For reproducible compilation, the synthesizer can be configured with a JSON file that describes the target device architecture.

This file is intended to be:

* explicit (no guessing)
* stable in CI/build systems
* extensible over time

> Mock example: `ada_mock.json`

```json
{
  "device": {
    "type": "gpu",
    "api": "vulkan",
    "vendor": "nvidia",
    "name": "Mock RTX",
    "architecture": "ada_lovelace",
    "driver": "555.xx"
  },
  "limits": {
    "max_workgroup_size": [1024, 1024, 64],
    "max_shared_memory_bytes": 65536,
    "max_push_constants_bytes": 256,
    "max_storage_buffer_range_bytes": 2147483647
  },
  "features": {
    "fp16": true,
    "int8": true,
    "subgroup_ops": true,
    "cooperative_matrix": false
  },
  "memory": {
    "global_bytes": 17179869184,
    "shared_bytes_per_sm": 65536,
    "l2_bytes": 67108864
  },
  "preferences": {
    "default_precision": "fp16",
    "prefer_fusion": true,
    "prefer_persistent_kv": true,
    "max_kernel_ops": 12
  }
}
```

#### Loading the device JSON

```rust
use openinfer::{Synthesizer, DeviceCustom};

let txt = std::fs::read_to_string("devices/ada_mock.json")?;
let arch: DeviceCustom = serde_json::from_str(&txt)?;

let synth = Synthesizer::from_arch(arch);
let plan = synth.synthesize(&model, &g)?;
plan.emit("build/out")?;
```

> The device JSON is how you make compilation **repeatable** across machines and CI environments.
