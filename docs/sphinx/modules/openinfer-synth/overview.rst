Overview
========

`openinfer-synth` (planned) will lower validated graphs into static, device-specific
source code for edge targets. It bridges the simulator graph to vendor
toolchains and hardware SDKs without introducing heavyweight runtimes.

Topics
------

- Planned scope and structure

Roadmap
-------

- Synthesis passes for scheduling and memory layout.
- Backend-specific code generation (C, CUDA, Vulkan, VHDL/HLS).
- Thin HAL bindings for constrained devices.
- Target focus: ARM + NEON, x86 + AVX, Vulkan GPUs, NVIDIA Jetson CUDA,
  Android NNAPI, USB TPUs (Coral), bare-metal MCUs, and FPGA flows.

