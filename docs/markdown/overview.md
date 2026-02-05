# Overview

## Core Idea

1. **Models live in a single binary package**

   * Weights and tensors
   * Constants and metadata
   * Shapes, dtypes, layouts
   * Optional persistent buffer definitions

2. **Developers write the execution logic**

   * Inputs and outputs
   * Control flow (loops, branches, yields)
   * Explicit operations
   * Explicit persistent memory access

3. **The result is a symbolic graph**

   * Nothing executes when defined
   * The DSL produces a structured graph of blocks and operations

4. **That graph can be**

   * Simulated for correctness on CPU or Vulkan
   * Traced and serialized for inspection
   * Analyzed and optimized (planned)

## Mental Model

Think of OpenInfer as a small, explicit IR embedded in Rust:

> **You describe what happens and in what order.**

The runtime executes the graph deterministically and makes tracing/debugging easy.
Longer-term analysis and synthesis passes are planned but not implemented yet.

## High-Level Workflow

```
Model Package (.oinf)
        │
        ▼
 graph! DSL  ──▶  Graph (blocks + nodes)
                        │
                        ├─▶ Simulator / Executor (CPU | Vulkan)
                        │
                        ├─▶ Trace + JSON serialization
                        │
                        └─▶ Analyzer / Synthesizer (planned)
```

For more information see [docs/implementation.md](implementation.md).

See [docs/types.md](types.md) for dtype support and Vulkan fallback behavior.
