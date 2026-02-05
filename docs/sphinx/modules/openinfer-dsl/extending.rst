Extending
=========

How to extend the DSL with new constructs.

Topics
------

- Adding a new keyword
- Updating codegen behavior

Steps
-----

1. Define a new keyword in the `kw` module.
2. Extend the parser to recognize the new syntax.
3. Map the AST to a `NodeKind` or variable declaration.
4. Add tests in `tests/openinfer-dsl`.
