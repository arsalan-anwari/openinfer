Overview
========

The `openinfer-dsl` crate provides the `graph!` macro for building graphs with a
compact syntax that feeds the simulator and synthesizer pipeline.

Topics
------

- DSL purpose and scope
- Integration with `openinfer`

Design philosophy
-----------------

- Explicit control flow and op ordering.
- Declarative variable definitions with memory kinds.
- No implicit optimizations or operator fusion.
- The DSL is a front-end to simulation and synthesis, not a runtime.

Structure
---------

The DSL has two top-level sections:

- Memory sections: `dynamic`, `volatile`, `constant`, `persistent`
- Blocks: `block entry { ... }`

Example
-------

.. code-block:: rust

   let g = graph! {
     dynamic {
       x: f32[B, D];
     }
     constant {
       W: f32[D, D];
     }
     volatile {
       y: f32[B, D];
     }

     block entry {
       op matmul(x, W) >> y;
       return;
     }
   };

This expands into calls that construct a `Graph`, allocate variables, and add
nodes to the `entry` block. The runtime executes the graph exactly as declared.


