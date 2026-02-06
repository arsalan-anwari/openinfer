Internals
=========

Parsing pipeline and code generation details.

Topics
------

- Tokenization and AST
- Validation steps

Parsing
-------

The macro uses `syn` to parse DSL tokens into a structured AST. Memory
declarations become `VarDecl` records and block nodes become `NodeKind` values.

Expansion
---------

The AST expands into Rust code that calls:

- `Graph::new()`
- `Graph::add_var(...)`
- `Graph::add_block(...)`
- `Graph::add_node(...)`

Validation
----------

Invalid syntax, unknown attributes, or invalid node forms are rejected at
compile time by the macro.


