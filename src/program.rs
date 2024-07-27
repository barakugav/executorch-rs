use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr;

use crate::evalue::EValue;
use crate::util::IntoRust;
use crate::{et_c, et_rs_c, DataLoader, MemoryManager, Result};

pub struct Program<'a>(et_c::Program, PhantomData<&'a ()>);
impl<'a> Program<'a> {
    pub fn load(
        data_loader: &'a impl DataLoader,
        verification: ProgramVerification,
    ) -> Result<Self> {
        let mut data_loader = data_loader.data_loader();
        Ok(Self(
            unsafe { et_c::Program::load(&mut *data_loader, verification) }.rs()?,
            PhantomData,
        ))
    }

    pub fn method_meta(&self, method_name: &str) -> Result<MethodMeta> {
        let method_name = CString::new(method_name).unwrap();
        let meta = unsafe { et_rs_c::Program_method_meta(&self.0, method_name.as_ptr()) }.rs()?;
        Ok(MethodMeta::new(meta))
    }

    pub fn load_method(
        &self,
        method_name: &str,
        memory_manager: &'a MemoryManager,
    ) -> Result<Method> {
        let method_name = CString::new(method_name).unwrap();
        let mut memory_manager = memory_manager.0.borrow_mut();
        let method = unsafe {
            self.0
                .load_method(method_name.as_ptr(), &mut *memory_manager, ptr::null_mut())
        };
        Ok(Method(method.rs()?))
    }
}

pub type ProgramVerification = et_c::Program_Verification;

pub struct MethodMeta(et_c::MethodMeta);
impl MethodMeta {
    fn new(method_meta: et_c::MethodMeta) -> Self {
        Self(method_meta)
    }

    pub fn num_memory_planned_buffers(&self) -> usize {
        unsafe { self.0.num_memory_planned_buffers() }
    }

    pub fn memory_planned_buffer_size(&self, idx: usize) -> Result<usize> {
        unsafe { et_rs_c::MethodMeta_memory_planned_buffer_size(&self.0, idx) }
            .rs()
            .map(|v| v as usize)
    }
}

pub struct Method(et_c::Method);
impl Method {
    pub fn start_execution(&mut self) -> Execution {
        Execution::new(self)
    }

    pub fn inputs_size(&self) -> usize {
        unsafe { self.0.inputs_size() }
    }
}

pub struct Execution<'a> {
    method: &'a mut Method,
    set_inputs: u64,
}
impl<'a> Execution<'a> {
    fn new(method: &'a mut Method) -> Self {
        assert!(
            method.inputs_size() <= u64::BITS as usize,
            "more that 64 inputs for method, unsupported"
        );
        Self {
            method,
            set_inputs: 0,
        }
    }

    pub fn set_input(&mut self, input: &'a EValue<'a>, input_idx: usize) -> Result<()> {
        unsafe { self.method.0.set_input(&input.0, input_idx) }.rs()?;
        self.set_inputs |= 1 << input_idx;
        Ok(())
    }

    pub fn execute(self) -> Result<Outputs<'a>> {
        assert_eq!(
            self.set_inputs,
            (1 << self.method.inputs_size()) - 1,
            "some inputs were not set"
        );
        unsafe { self.method.0.execute() }.rs()?;
        Ok(Outputs::new(self.method))
    }
}
pub struct Outputs<'a> {
    method: &'a mut Method,
}
impl<'a> Outputs<'a> {
    fn new(method: &'a mut Method) -> Self {
        Self { method }
    }

    pub fn get_output(&self, output_idx: usize) -> &'a EValue<'a> {
        let val = unsafe { self.method.0.get_output(output_idx) };
        // SAFETY: et_c::EValue as EValue has the same memory layout
        let ptr = val as *const EValue;
        unsafe { &*ptr }
    }
}
