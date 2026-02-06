Serialization
=============

Graphs are plain Rust objects and can be serialized to JSON.

Serialize
---------

.. code-block:: rust

   let json = GraphSerialize::json(&g)?;
   std::fs::write("graph.json", serde_json::to_string_pretty(&json)?)?;

Deserialize
-----------

.. code-block:: rust

   let value = serde_json::from_str(&graph_txt)?;
   let g = GraphDeserialize::from_json(value)?;
