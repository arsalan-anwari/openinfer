# Tensors

OpenInfer tensors store:

- data (flat `Vec<T>`)
- shape (rank and dimension sizes)
- strides (indexing layout)

The primary type is `Tensor<T>` in `openinfer/src/tensor.rs`.

## Creating Tensors

Basic vector tensor (1D):

```rust
use openinfer::tensor::Tensor;

let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0])?;
```

Explicit shape and options:

```rust
use openinfer::tensor::{Tensor, TensorOptions};

let t = Tensor::from_vec_with_opts(
    vec![1.0f32, 2.0, 3.0, 4.0],
    TensorOptions {
        shape: Some(vec![2, 2]),
        ..TensorOptions::default()
    },
)?;
```

Scalar tensor:

```rust
use openinfer::tensor::Tensor;

let t = Tensor::from_scalar(3.14f32);
```

## Shape and Strides

```rust
let shape = t.shape();
let strides = t.strides();
let len = t.len();
```

By default, strides are contiguous. You can supply custom strides via
`TensorOptions`, but shape and stride lengths must match.

## Indexing and Views

You can index with a fixed-size array to get a view:

```rust
let row = t[[1]].to_vec();
```

```rust
let row = t.at(&[1]);
```

Or single element reference:

```rust
let row = t[[1, 0]].to_vec();
```

```rust
let value = t.at(&[1, 0]);
```

Or whole flat array
```rust
let raw = t.to_vec();
```

Views are lightweight and share the underlying data.

## TensorValue

`TensorValue` is an enum wrapper for typed tensors:

```rust
use openinfer::tensor::TensorValue;

let value = TensorValue::from(42u64);
```

Use `TensorValue::dtype`, `TensorValue::shape`, and `TensorValue::len` to inspect
the stored tensor.
