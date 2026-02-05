# Graph (De)Serialization

Graphs are plain Rust objects and can be serialized to JSON.

Serialization:
```rust
let json = GraphSerialize::json(&g)?;
std::fs::write("minimal-graph.json", serde_json::to_string_pretty(&json)?)?;
```

Deserialization:
```rust
let value = serde_json::from_str(&graph_txt)?;
let g = GraphDeserialize::from_json(value)?;
```
