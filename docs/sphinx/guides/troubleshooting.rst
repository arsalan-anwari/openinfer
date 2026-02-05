Troubleshooting
===============

Common build and runtime errors with fixes.

Build failures
--------------

- Missing Rust toolchain: install via `rustup`.
- Missing Python deps: run `pip install -r docs/sphinx/requirements.txt`.
- Vulkan build errors: ensure `--features vulkan` and a Vulkan SDK.

Runtime validation errors
-------------------------

- Graph variable dims do not match `.oinf` sizevars.
- DSL variable name conflicts with model variable of a different dtype.

Vulkan fallback
---------------

If a GPU does not support `shader_int64` or `shader_float64`, Vulkan ops will
fall back to CPU and emit a warning. This is expected behavior for unsupported
types.

Serialization issues
--------------------

- `.oinf` file corruption or invalid offsets trigger `ModelLoader::open` errors.
- Use the Python verifier to inspect format issues.
