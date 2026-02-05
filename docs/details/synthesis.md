# Backend Documentation

See backend-specific docs for implementation details and extension points:

- [docs/vulkan-interop.md](vulkan-interop.md)

## Supported Ops

For a current list of supported ops and per-backend dtype coverage, see:

- [docs/ops.md](ops.md)

## Synthesis (Planned)

OpenInfer does not currently ship a Synthesizer or device-architecture JSON
inputs. The runtime today is the Simulator/Executor, which runs graphs on CPU
or Vulkan with optional tracing.

### Current alternatives

- Use the simulator for correctness checks and tracing.
- Use graph JSON serialization (`GraphSerialize`/`GraphDeserialize`) for tooling
  pipelines while the synthesis stage is still under development.
