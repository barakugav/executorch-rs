#![deny(warnings)]

mod c_link;
use c_link::executorch_c::root::executorch_rs as et_rs_c;
use c_link::executorch_c::root::torch::executor as et_c;

mod error;
pub use error::{Error, Result};

pub mod data_loader;

mod memory;
pub use memory::{HierarchicalAllocator, MallocMemoryAllocator, MemoryManager};

mod program;
pub use program::{Method, MethodMeta, Program, ProgramVerification};

#[cfg(feature = "extension-module")]
mod module;
#[cfg(feature = "extension-module")]
pub use module::Module;

mod evalue;
pub use evalue::{EValue, Tag};

mod tensor;
pub use tensor::{Tensor, TensorImpl, TensorInfo, TensorMut};

mod util;
pub use util::{ArrayRef, Span};

pub fn pal_init() {
    unsafe { c_link::executorch_c::root::et_pal_init() };
}
