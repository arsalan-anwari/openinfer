Add a DSL Node
==============

Walkthrough for extending the graph DSL with a new construct.

Outline
-------

- Update parser keywords
- Extend AST and validation
- Regenerate tests

1. Add a keyword
---------------------

Define a new keyword in `openinfer-dsl/src/lib.rs` under `mod kw`.

2. Parse the new node
---------------------

Update the parser to recognize the syntax and emit a new AST node.

.. code-block:: rust

   block entry {
     my_node x, y >> z;
   }

3. Map to `NodeKind`
--------------------

Extend expansion logic to map the new AST node to a `NodeKind` variant or a
new node type if needed.

4. Validate
-----------

Add validation rules to reject invalid forms early in macro expansion.

5. Tests
--------

Add parser tests under `openinfer-dsl/tests/parse_tests.rs`.
