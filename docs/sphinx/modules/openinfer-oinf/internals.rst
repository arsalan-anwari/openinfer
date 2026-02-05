Internals
=========

Core implementation details of the `.oinf` encoder and verifier.

Topics
------

- Format layout and offsets
- Packed dtype handling

File layout
-----------

The `.oinf` file contains:

- Header (magic, version, offsets).
- Sizevars table.
- Metadata table.
- Tensor table.
- Payload data section.

Packed types
------------

Packed integer dtypes are stored bit-packed in payloads. The Python verifier
unpacks these into NumPy arrays for inspection.

Rust loader
-----------

`ModelLoader::open` memory-maps the file, validates header tables, and loads
tensor payloads lazily when requested by the executor.
