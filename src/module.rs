use std::path::Path;

use crate::util;
use crate::{et_c, et_rs_c, util::IntoRust, EValue, Result, Span};

pub struct Module(et_c::Module);
impl Module {
    pub fn new(file_path: impl AsRef<Path>) -> Self {
        let file_path = file_path.as_ref().to_str().unwrap();
        let file_path = Span::new(util::str2chars(file_path).unwrap());
        Self(unsafe { et_rs_c::Module_new(file_path.0) })
    }

    pub fn forward<'a>(&'a mut self, inputs: &[EValue]) -> Result<Vec<EValue<'a>>> {
        let method_name = Span::new(util::str2chars("forward").unwrap());
        // Safety: The transmute is safe because the memory layout of EValue and et_c::EValue is the same.
        let inputs = unsafe { std::mem::transmute::<&[EValue], &[et_c::EValue]>(inputs) };
        let inputs = Span::new(inputs);
        let outputs =
            unsafe { et_rs_c::Module_execute(&mut self.0, method_name.0, inputs.0) }.rs()?;
        let outputs = outputs.rs();
        // Safety: The transmute is safe because the memory layout of EValue and et_c::EValue is the same.
        let outputs = unsafe { std::mem::transmute::<Vec<et_c::EValue>, Vec<EValue<'a>>>(outputs) };
        Ok(outputs)
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        unsafe { et_rs_c::Module_destructor(&mut self.0) };
    }
}
