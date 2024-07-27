use std::ffi::CString;
use std::path::Path;

use crate::{et_c, et_rs_c, util::IntoRust, EValue, Result};

#[allow(dead_code)]
pub struct Module(et_c::Module);
impl Module {
    pub fn new(file_path: impl AsRef<Path>) -> Self {
        let method_name = CString::new(file_path.as_ref().to_str().unwrap()).unwrap();
        Self(unsafe { et_rs_c::Module_new(method_name.as_ptr()) })
    }

    pub fn forward<'a>(&'a mut self, inputs: &[EValue]) -> Result<Vec<EValue<'a>>> {
        let method_name = CString::new("forward").unwrap();
        // Safety: The transmute is safe because the memory layout of EValue and et_c::EValue is the same.
        let inputs = unsafe { std::mem::transmute::<&[EValue], &[et_c::EValue]>(inputs) };
        let outputs = unsafe {
            et_rs_c::Module_execute(
                &mut self.0,
                method_name.as_ptr(),
                inputs.as_ptr(),
                inputs.len(),
            )
        }
        .rs()?;
        let outputs = outputs.rs();
        // Safety: The transmute is safe because the memory layout of EValue and et_c::EValue is the same.
        let outputs = unsafe { std::mem::transmute::<Vec<et_c::EValue>, Vec<EValue<'a>>>(outputs) };
        Ok(outputs)
    }
}
