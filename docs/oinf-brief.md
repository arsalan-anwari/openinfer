# OINF Binary Format (Brief)

OpenInfer models are stored in a single `.oinf` binary container that holds:

* Named size variables (u64 only)
* Metadata key/value pairs
* Named tensors with dtype, shape, and optional data

Conceptually, an uncompacted view looks like:

```ini
B := 1024
a: f32[B] = { ... }
mode: str = "clamp_up"
```

Create a binary file:

```bash
python examples/openinfer-oinf/minimal_oinf.py
```

> See [docs/oinf.md](oinf.md) how you can create your own binary file from a Python Dataclass.

Inspect it:

```bash
python openinfer-oinf/verify_oinf.py res/models/minimal_model.oinf
```

For the full technical spec and layout, see [docs/oinf.md](oinf.md).
