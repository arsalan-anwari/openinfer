# FAQ

## What is OpenInfer for?

OpenInfer is for describing and executing inference graphs with explicit control
flow, predictable behavior, and a clear separation between model data and logic.

## Do I need Vulkan?

No. CPU execution is always available and is the reference path. Vulkan is
optional and enabled with `--features vulkan`.

## How do I create a model file?

Use the Python tooling under `openinfer-oinf/`, for example:

```bash
python examples/openinfer-oinf/mlp_regression_oinf.py
```

This writes a `.oinf` file in `res/models/`.

## Can I use my own model weights?

Yes. `.oinf` is a simple binary container for sizevars, metadata, and tensors.
You can generate it with the Python dataclass helper or your own tooling.

## Why are CPU and GPU results slightly different?

Low‑precision formats and GPU execution order can introduce small drift. This
is expected for types like `f8`, `bf16`, and `f16`.
