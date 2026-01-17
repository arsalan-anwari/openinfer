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

   * Simulated for correctness
   * Analyzed and optimized
   * Compiled into device-specific code

## Mental Model

Think of OpenInfer as:

> **A small, ML-focused Synthesizer frontend embedded in Rust.**

You describe **what happens and in what order**.
OpenInfer decides **how to execute it efficiently**.

The DSL is closer in spirit to **ONNX / XLA / TVM IRs** than to eager frameworks like PyTorch.

## High-Level Workflow

```
Model Package (.oinf)
        │
        ▼
 graph! DSL  ──▶  Symbolic Graph
                        │
                        ├─▶ Simulator (correctness-first)
                        │
                        └─▶ Analyzer / Synthesizer
                               ▼
                        Device-specific source code
```

For more information see [docs/implementation.md](implementation.md)
