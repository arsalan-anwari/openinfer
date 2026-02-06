Overview
========

`openinfer-oinf` provides Python encoders and verifiers for the `.oinf` format,
which supplies model weights and metadata to the simulator and synthesizer.

Topics
------

- Format goals and scope
- Tooling entry points

Format goals
------------

- Single-file binary package for model weights and metadata.
- Deterministic layout and offsets.
- Lazy loading in the Rust runtime.

Tooling entry points
--------------------

- `dataclass_to_oinf.py`: encode Python dataclasses to `.oinf`.
- `verify_oinf.py`: verify and pretty-print `.oinf` files.


