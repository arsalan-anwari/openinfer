Internals
=========

The generator crate is organized into:

- Parsers: load JSON/TOML inputs into typed structs.
- Renderers: translate structured data into text outputs.
- Helpers: stable ordering, naming, and formatting utilities.

The design goal is minimal dependency on the runtime. This keeps the generator
fast and decoupled from the core OpenInfer runtime.

Input validation
----------------

Inputs are validated early so failures are caught close to the source. For
example, an op entry with a missing `name` or malformed dtype list should fail
the generator, not the runtime.

Rendering strategy
------------------

Renderers build output using small templates and explicit formatting rules. This
keeps the output consistent across platforms and avoids diff noise in generated
files.
