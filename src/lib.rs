#![deny(warnings)]

use executorch_sys::executorch_rs as et_rs_c;
use executorch_sys::torch::executor as et_c;

#[macro_use]
mod private;
pub mod data_loader;
pub mod error;
pub mod evalue;
pub mod memory;
#[cfg(feature = "module")]
pub mod module;
pub mod program;
pub mod tensor;
pub mod util;

pub fn pal_init() {
    unsafe { executorch_sys::et_pal_init() };
}
