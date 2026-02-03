use anyhow::Result;
use openinfer::{Tensor, TensorOptions};

#[test]
fn tensor_strides_match_default_layout() -> Result<()> {
    let data = vec![0.0f32; 6];
    let tensor = Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(vec![2, 3]),
            ..TensorOptions::default()
        },
    )?;
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.strides(), &[3, 1]);
    Ok(())
}

#[test]
fn tensor_allows_len_mismatch_for_packed_shapes() -> Result<()> {
    let data = vec![1u8; 2];
    let tensor = Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(vec![3]),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )?;
    assert_eq!(tensor.shape(), &[3]);
    Ok(())
}
