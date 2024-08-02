mod c_link {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]

    include!(concat!(env!("OUT_DIR"), "/executorch_bindings.rs"));
}
pub use c_link::root::*;

use crate::executorch_rs as et_rs_c;
use crate::torch::executor as et_c;

impl Drop for et_c::Tensor {
    fn drop(&mut self) {
        unsafe { et_rs_c::Tensor_destructor(self) }
    }
}
