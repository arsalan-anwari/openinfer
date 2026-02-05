Extending
=========

When adding a new generator:

1. Define a typed input schema.
2. Add validation rules for required fields.
3. Render output with stable ordering.
4. Update scripts to invoke the generator.

Prefer making the generator idempotent so repeated runs do not change output.

Testing a generator
-------------------

For new generators, add a small fixture input and compare the generated output
with a golden file. This keeps the generator stable across refactors.
