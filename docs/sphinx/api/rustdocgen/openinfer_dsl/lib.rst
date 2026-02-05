=======================
Crate ``openinfer_dsl``
=======================

.. rust:crate:: openinfer_dsl
   :index: 0

   Procedural macro DSL for building OpenInfer graphs.

   The `graph!` macro parses a compact DSL into `openinfer::Graph` structures.
   It is intended for ergonomics in tests and examples.

   .. rust:use:: openinfer_dsl
      :used_name: self

   .. rust:use:: openinfer_dsl
      :used_name: crate

   .. rust:use:: proc_macro::TokenStream
      :used_name: TokenStream

   .. rust:use:: openinfer_dsl::types::GraphDsl
      :used_name: GraphDsl

   .. rubric:: Functions

   .. rust:function:: openinfer_dsl::graph
      :index: 0
      :vis: pub
      :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"graph"},{"type":"punctuation","value":"("},{"type":"name","value":"input"},{"type":"punctuation","value":": "},{"type":"link","value":"TokenStream","target":"TokenStream"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"TokenStream","target":"TokenStream"}]

      Build an OpenInfer `Graph` from the DSL input.
