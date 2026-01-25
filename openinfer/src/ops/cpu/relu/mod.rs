mod relu;
mod relu_accumulate;
mod relu_inplace;
pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

pub use relu::*;
#[allow(unused_imports)]
pub use relu_accumulate::*;
#[allow(unused_imports)]
pub use relu_inplace::*;

#[allow(dead_code)]
pub fn supports_broadcast() -> bool {
    false
}
