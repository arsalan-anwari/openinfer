Synthesizer Context and Roadmap
===============================

This chapter explains the role of the Synthesizer in the OpenInfer roadmap,
what it is intended to do, and how you should think about it when extending the
codebase today. The Synthesizer is not fully implemented in the current tree,
but it remains an important architectural concept: it is a *separate* codegen
pipeline that inspects a graph and generates device-optimized source code with
a hardware abstraction layer (HAL) in C, VHDL, or other targets.

The absence of a full Synthesizer does not mean the project lacks structure.
OpenInfer already has a Simulator and Executor that run graphs deterministically
on CPU or Vulkan. The Synthesizer is *separate* from the executor: it does not
interpret the graph at runtime. Instead, it analyzes the graph and transpiles
it into a sequence of calls to a prebuilt C/VHDL/etc library, producing a
standalone, device-specific implementation.

Why a Synthesizer exists
------------------------

Today, OpenInfer executes the graph you provide. It does not insert implicit
optimizations, does not fuse ops, and does not reorder nodes across barriers.
This is a deliberate design choice: correctness and explicit control are
prioritized over automatic optimization.

However, some workloads benefit from a separate synthesis stage:

- Graph-level transformations (fusion, reordering, common subexpression reuse).
- Device-specific code generation (C/VHDL HAL targets).
- Memory planning (pre-allocation and reuse strategies for the generated code).
- Cost modeling and auto-tuning.

These are exactly the responsibilities of a Synthesizer. The Synthesizer is
planned as the place where you *opt in* to optimization and codegen, rather than
having it hidden inside the executor. This keeps the explicit nature of the DSL
while allowing advanced performance work later.

Current alternatives
--------------------

Since the Synthesizer is not fully implemented, you should use the existing
tools to achieve similar goals:

- Use the **Simulator** for validation and trace capture.
- Use **Graph serialization** to build tooling around graph transformation.
- Use **explicit DSL structure** to encode control flow and memory planning.

Graph serialization is particularly important. The graph can be serialized to
JSON, which means you can build external tools that read, transform, and
re-emit graphs. This is a practical way to experiment with synthesis-like
behavior today without modifying the runtime.

In other words, the current workflow is:

1. Author a graph with the DSL.
2. Validate and execute it with the simulator/executor.
3. Optionally transform the graph using external tooling (JSON).
4. Re-run the transformed graph through the same pipeline.

This keeps experimentation outside the runtime while preserving deterministic
execution.

Planned responsibilities
------------------------

The Synthesizer is expected to handle several layers of optimization and
code generation. The exact implementation may evolve, but the design intent is
stable:

**Graph analysis**
  - Identify independent subgraphs.
  - Detect opportunities for op fusion.
  - Analyze control-flow structure for scheduling opportunities.

**Device-aware codegen**
  - Select a HAL target (C, VHDL, etc.) and compatible libraries.
  - Emit a sequence of function calls into a prebuilt HAL/runtime library.
  - Respect dtype support and device constraints in the generated code.

**Memory planning**
  - Allocate buffers up front when possible.
  - Reuse temporary buffers across non-overlapping lifetimes.
  - Manage cache table growth and lifecycle explicitly.

**Cost modeling**
  - Estimate runtime cost for each op or block.
  - Use heuristics to pick between candidate plans.
  - Provide trace annotations for performance analysis.

The key is that these responsibilities are separated from the executor. The
executor remains a deterministic interpreter of the graph; the Synthesizer
produces device-specific source code (or artifacts) that can run without the
executor.

How to design code today with synthesis in mind
-----------------------------------------------

Even though the Synthesizer is not implemented, you can write code that aligns
with the eventual architecture:

- Keep graphs explicit and structured.
- Use blocks to delineate logical phases (e.g., pre-processing, main compute).
- Use cache operations explicitly so memory state is visible.
- Avoid embedding optimization logic directly in kernels.

This makes it easier to insert a synthesis stage later. If your graph already
encodes explicit dependencies and state transitions, a future Synthesizer can
analyze it without guesswork.

When adding ops, prefer to keep kernels simple and composable. Complex fused
ops can be added later by a Synthesizer if there is demand. The current design
expects op fusion and similar optimizations to happen *outside* the core op
library.

Potential interaction with OINF and DSL
---------------------------------------

The Synthesizer is expected to sit alongside the executor as an alternative
path:

1. DSL builds a graph.
2. Executor interprets the graph *or* the Synthesizer transpiles it.
3. The Synthesizer emits device-optimized source code (C/VHDL/etc) that calls
   into a HAL/runtime library.

The OINF file remains the model source of truth. The Synthesizer should not
change the model parameters; it should only change execution structure. This
keeps data integrity intact and makes it easier to validate results.

In some designs, the Synthesizer might also generate device-specific graphs or
multiple plan variants. In that case, the toolchain could select a plan based on
device capabilities or performance goals. This is still an open design space,
but the core principle is that the Synthesizer is a *codegen layer*, not a data
authoring layer.

Tracing and synthesis
---------------------

One reason the Synthesizer is attractive is that OpenInfer already has tracing.
The trace includes node timing and metadata. A future Synthesizer could use
this data to refine its cost model or to validate that its transformations are
beneficial.

For example:

- Run a graph with tracing enabled.
- Collect timing per node and per op.
- Identify hotspots and candidate fusion groups.
- Generate a new graph with fused ops.
- Re-run and compare traces.

This workflow is possible today with external tooling because the trace is a
structured JSON artifact. The Synthesizer can eventually internalize this loop.

Relationship to openinfer-synth
-------------------------------

The `openinfer-synth` module is the placeholder for synthesis-related tooling.
Even though it currently contains minimal content, it is the natural place for
future synthesis infrastructure, including:

- Graph transformation utilities.
- Cost modeling experiments.
- Device capability descriptions.

If you plan to contribute to synthesis features, keep code modular and avoid
coupling it to the runtime. The goal is to allow experimentation without
destabilizing the core executor.

Practical guidance for contributors
-----------------------------------

If you are contributing today, here is the most useful mindset:

- Treat synthesis as a future optimization layer, not a requirement.
- Focus on correctness and explicitness in the DSL and runtime.
- Keep ops and kernels predictable and easy to compose.
- Use graph serialization and external tooling when you need transformation.

This matches the existing architecture and keeps your changes aligned with the
long-term plan.

Where to go next
----------------

- `Architecture Overview` for the current execution pipeline.
- `Modules/openinfer-synth` for the synthesis module overview.
- `Serialization` for how to transform graphs externally today.

Potential synthesis artifacts
-----------------------------

A future Synthesizer may produce artifacts in addition to a transformed graph.
These artifacts are useful for debugging and for reproducibility:

- **Plan metadata**: a summary of device choices and HAL targets.
- **Memory plan**: a list of buffer allocations and reuse decisions.
- **Codegen output**: generated C/VHDL/etc sources and build instructions.
- **Trace annotations**: mappings from original nodes to synthesized code.

The exact format is still open, but a JSON or structured binary format would be
natural given the existing graph serialization.

Example transformation concept
------------------------------

To make synthesis more concrete, consider a simple pattern:

.. code-block:: text

   op add(x, y) >> t0
   op relu(t0) >> t1

A Synthesizer could detect that `add` and `relu` can be fused into a single
kernel on a target device. It would then emit code that calls a fused HAL
function instead of two separate calls. Importantly, the transformation is
explicit in the generated artifacts, even though the executor is not involved.

The advantage is that the runtime stays simple. The fused op is still a normal
op from the executor’s perspective; the difference is that the Synthesizer
generated it.

More examples that are specific to device codegen:

- **Memory-limit tiling / streaming**: if a tensor does not fit in device memory,
  the Synthesizer can split the graph into multiple smaller streams, emitting a
  loop in the generated C/VHDL that processes tiles sequentially.
- **Async chunked I/O**: for large tensor transfers, the Synthesizer can emit
  async load/store calls and overlap DMA with compute using a double-buffered
  schedule.
- **Target-specific kernel selection**: emit vectorized GPU kernels, TPU
  systolic-array friendly layouts, or FPGA pipeline stages based on the target.
- **Bandwidth-aware scheduling**: reorder independent nodes to hide memory
  latency or batch small ops into a single hardware call.

Concrete sketch (streaming + async copies):

.. code-block:: text

   // Pseudocode emitted by the Synthesizer
   for tile in tiles(A, tile_size) {
     hal_async_load(A_tile[tile], device_buf0);
     hal_async_load(B_tile[tile], device_buf1);
     hal_wait(device_buf0, device_buf1);
     hal_matmul(device_buf0, device_buf1, device_out);
     hal_async_store(device_out, C_tile[tile]);
   }

These transformations are not implicit runtime magic; they are explicit in the
generated artifacts and can be inspected or profiled per target.

Open questions and design constraints
-------------------------------------

There are still open questions in the synthesis design:

- Should the Synthesizer produce one plan or multiple alternatives?
- How should it express device fallback when a kernel is missing?
- How should it represent transformations that change control flow?
- How should it interact with cache tables and persistent state?

The existing architecture leans toward explicit, inspectable artifacts. Whatever
design emerges should preserve that property. That is why the project currently
emphasizes explicit graphs and traceability: it sets a foundation that synthesis
can build on without losing debuggability.
