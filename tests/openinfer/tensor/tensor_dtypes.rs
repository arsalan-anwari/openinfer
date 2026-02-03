use openinfer::DType;

#[test]
fn dtype_float_flags() {
    assert!(DType::F32.is_float());
    assert!(DType::F16.is_float());
    assert!(DType::BF16.is_float());
    assert!(DType::F8.is_float());
    assert!(!DType::I32.is_float());
}

#[test]
fn dtype_packed_storage_len() {
    let logical = 9usize;
    assert_eq!(DType::I1.storage_len(logical), 2);
    assert_eq!(DType::U1.storage_len(logical), 2);
    assert_eq!(DType::I2.storage_len(logical), 3);
    assert_eq!(DType::U2.storage_len(logical), 3);
    assert_eq!(DType::I4.storage_len(logical), 5);
}
