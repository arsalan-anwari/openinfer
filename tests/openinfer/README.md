# Openinfer Tests

This directory contains integration tests for the `openinfer` crate, plus
numpy-driven baselines stored as `.oinf` binaries. The ops suite covers
elementwise ops, broadcasts, comparisons, bitwise ops, rounding, reductions,
casts, accumulate/inplace paths, and misc ops like fma/floor_div/is_finite.
Graph suites cover branching, cache behavior, loops, yield/await, prefix tables,
reference vars, attribute loading, serialization, and tracing.
Packed dtypes (i4/u2) and custom floats (f16/bf16/f8) are covered in dedicated
ops suites.
The full-matrix ops runner uses `tests/openinfer/ops/baseline/data/full_matrix/manifest.json`
and `inventory.json` to cover every op/dtype/mode supported by the schema.

## Baseline generation

Regenerate all numpy baselines:

```
python tests/openinfer/ops/baseline/gen_ops_baseline.py
python tests/openinfer/graph/baseline/gen_graph_baseline.py
```

The scripts write `.oinf` files and a JSON manifest alongside them under
`tests/openinfer/*/baseline/data`.
Full-matrix baselines are written to `tests/openinfer/ops/baseline/data/full_matrix`.
`bitset` tensors are currently skipped in the full-matrix baselines because
`openinfer-oinf` only supports `bitset` as metadata today.

## Running tests

Run all `openinfer` tests:

```
cargo test -p openinfer
```

Run just the openinfer test suite:

```
cargo test -p openinfer --test openinfer
```

### Device selection

By default tests run on CPU. To try Vulkan as well, set:

```
TEST_TARGETS=cpu,vulkan cargo test -p openinfer --test openinfer
```

If Vulkan is unavailable on the machine, the Vulkan runs will be skipped with a
message and the CPU run will still validate the baseline.

Use the `TEST_TARGETS` environment variable to select devices.
