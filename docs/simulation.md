# Simulation

Simulation mode:

* Executes lazily
* Loads model data on demand
* Preserves cache across runs
* Prioritizes correctness and debuggability

Simulation is designed to validate **logic and structure**, not raw performance.
The simulator validates the graph against the model at construction time (dtype
compatibility, constant mutation, scalar-only attributes, sizevar resolution),
and `make_executor()` reuses the validated graph.

By default, the simulator does not print trace output or time ops. Enable
`with_trace()` for trace logging and `with_timer()` for timing data. Use
`with_inplace()` to opt into in-place execution for supported ops when the
output aliases an input (e.g. `op add(x, y) >> x`), which is useful for
measuring allocation overhead vs. the standard out-of-place behavior.

## Step through nodes with logging + timing

```rust
use openinfer::{
  fetch_executor, format_truncated, graph, insert_executor,
  Device, ModelLoader, Simulator, Tensor
};

...

let sim = Simulator::new(&model, &g, Device::Cpu)?
  .with_trace()
  .with_timer();
let mut exec = sim.make_executor()?;
let input = Tensor::from_vec(input)?;
insert_executor!(exec, { x: input });

// This is equivalent to what happens when you call exec.step() with tracing enabled.
for mut node in exec.iterate() {
    let ev = node.event.clone();
    fetch_executor!(node, { y: Tensor<f32> });
    let y_str = format_truncated(&y.data);
    let y_pad = format!("{:<width$}", y_str, width = 32);
    println!(
        "y={} -- [{}] {} :: {} ({})",
        y_pad,
        ev.kind,
        ev.block_name,
        ev.op_name,
        ev.micros
    );
}
```

## Export a simulator trace

```rust
let sim = Simulator::new(&model, &g, Device::Cpu)?
  .with_trace()
  .with_timer();
let mut exec = sim.make_executor()?;
let input = Tensor::from_vec(input)?;
insert_executor!(exec, { x: input });
exec.run_step()?;
let trace = exec.trace();
std::fs::write("build/trace.json", serde_json::to_string_pretty(&trace)?)?;
```

Example of a trace output:

```json
[
  {
    "block_name": "entry",
    "node_index": 0,
    "node_uuid": "f4137d97-0919-4f73-9839-dd74367f943c",
    "kind": "Assign",
    "params": [],
    "output": ["t0"],
    "micros": [0, 0, 0]
  },
  {
    "block_name": "entry",
    "node_index": 1,
    "node_uuid": "53b7184e-4b0f-492d-8d93-9274d52617a6",
    "kind": "OpExecute",
    "params": ["x", "a"],
    "output": ["t0"],
    "micros": [0, 31, 788]
  },
  {
    "block_name": "entry",
    "node_index": 2,
    "node_uuid": "16ce2716-d6fc-4136-bd6f-1b8d5a66669e",
    "kind": "OpExecute",
    "params": ["y", "t0"],
    "output": ["y"],
    "micros": [0, 30, 12]
  },
  {
    "block_name": "entry",
    "node_index": 3,
    "node_uuid": "8500432b-f1c3-4d14-86de-af5c8f97506b",
    "kind": "Return",
    "params": [],
    "output": [],
    "micros": [0, 0, 0]
  }
]
```
