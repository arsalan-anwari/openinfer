Extending
=========

How to add new metadata fields or dtype support.

Topics
------

- Adding new ValueType entries
- Maintaining compatibility

Add a ValueType
---------------

1. Extend `ValueType` and mapping tables in `oinf_types.py`.
2. Update packing/unpacking logic if the type is packed.
3. Update verifier logic to parse and display the new type.
4. Add tests in `tests/openinfer-oinf`.
