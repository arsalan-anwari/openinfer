mod add;
mod add_accumulate;
mod add_inplace;
pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

pub use add::*;
#[allow(unused_imports)]
pub use add_accumulate::*;
pub use add_inplace::*;

#[allow(dead_code)]
pub fn supports_broadcast() -> bool {
    false
}
