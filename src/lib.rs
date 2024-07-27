#![deny(warnings)]

mod c_link;
use c_link::executorch_c::root::executorch_rs as et_rs_c;
use c_link::executorch_c::root::torch::executor as et_c;

mod error;
pub use error::{Error, Result};

mod data_loader;
pub use data_loader::DataLoader;
#[cfg(feature = "extension-data-loader")]
pub use data_loader::FileDataLoader;

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
pub use tensor::{Tensor, TensorImpl};

mod util;
pub use util::Span;

pub fn pal_init() {
    unsafe { c_link::executorch_c::root::et_pal_init() };
}
