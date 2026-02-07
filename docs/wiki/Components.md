# Components

OpenInfer is organized around a few core components.

## Rust DSL (graph!)

The `graph! {}` macro defines:

- Memory sections (`dynamic`, `volatile`, `constant`, `persistent`)
- Blocks with explicit nodes
- Control flow (loops, branches, yield/await)

The result is a symbolic graph that can be validated, serialized, and executed.

## Model loader

`ModelLoader` opens `.oinf` files and exposes:

- Size variables
- Metadata
- Tensor headers and lazy data loading

It defers reading tensor payloads until a node actually consumes them.

## Simulator + Executor

The simulator validates the graph and prepares execution. The executor:

- Loads inputs and constants
- Runs ops in order
- Updates persistent state
- Provides trace/timing output (optional)

## Ops registry

Ops are defined in `ops.json`:

- Op names and attributes
- Type rules and dtype coverage
- Broadcast/in‑place/accumulation support

CPU and Vulkan implementations are selected from this registry at runtime.

## Backends

- **CPU**: reference implementation focused on correctness
- **Vulkan**: optional GPU backend using precompiled SPIR‑V

## Synthesizer (planned)

The synthesizer lowers validated graphs into static, device-specific source
code for edge targets such as ARM + NEON, x86 + AVX, Vulkan GPUs, NVIDIA Jetson
CUDA, Android NNAPI, USB TPUs (Coral), bare-metal MCUs, and FPGA flows
(VHDL/HLS).

## Packages

- Rust crates: `openinfer-simulator`, `openinfer-dsl`, `openinfer-synth`
- Python tooling: `openinfer-oinf`
