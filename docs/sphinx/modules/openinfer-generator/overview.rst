Overview
========

`openinfer-generator` contains code-generation helpers that keep metadata and
schemas consistent across the project. These utilities are used during
development to render structured inputs (like `ops.json`) into code or docs.

Key responsibilities
--------------------

- Parse generator inputs into typed structures.
- Render deterministic outputs (stable ordering and formatting).
- Provide a single source of truth for op schemas.

Typical usage
-------------

The generator is invoked from scripts, not from the runtime. See the
`docs/sphinx/generate_ops_docs.py` flow for how `ops.json` is turned into the
Sphinx ops catalog.

Why a separate crate
--------------------

Keeping generation logic out of the runtime avoids pulling in extra dependencies
and makes build times smaller. It also makes it easier to evolve schemas without
touching execution logic.
