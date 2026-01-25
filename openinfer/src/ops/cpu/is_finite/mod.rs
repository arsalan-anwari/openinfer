mod is_finite;
mod is_finite_accumulate;
pub mod registry;
pub mod registry_accumulate;

pub use is_finite::*;

#[allow(dead_code)]
pub fn supports_broadcast() -> bool {
    false
}
#[allow(unused_imports)]
pub use is_finite_accumulate::*;
