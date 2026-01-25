mod fill;
mod fill_accumulate;
pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

pub use fill::*;
#[allow(unused_imports)]
pub use fill_accumulate::*;

#[allow(dead_code)]
pub fn supports_broadcast() -> bool {
    false
}
