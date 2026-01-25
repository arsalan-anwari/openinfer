mod abs;
mod abs_accumulate;
mod abs_inplace;
pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

pub use abs::*;
#[allow(unused_imports)]
pub use abs_accumulate::*;
#[allow(unused_imports)]
pub use abs_inplace::*;

#[allow(dead_code)]
pub fn supports_broadcast() -> bool {
    false
}
