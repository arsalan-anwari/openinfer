mod matmul;
mod matmul_accumulate;
mod matmul_inplace;
pub mod registry_accumulate;
pub mod registry_inplace;
pub mod registry;

pub use matmul::*;
#[allow(unused_imports)]
pub use matmul_accumulate::*;
#[allow(unused_imports)]
pub use matmul_inplace::*;

#[allow(dead_code)]
pub fn supports_broadcast() -> bool {
    true
}
