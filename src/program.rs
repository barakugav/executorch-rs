use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr;

use crate::evalue::EValue;
use crate::util::IntoRust;
use crate::{et_c, et_rs_c, DataLoader, MemoryManager, Result};

/// A deserialized ExecuTorch program binary.
pub struct Program<'a>(et_c::Program, PhantomData<&'a ()>);
impl<'a> Program<'a> {
    /// Loads a Program from the provided loader. The Program will hold a pointer
    /// to the loader, which must outlive the returned Program instance.
    ///
    /// # Arguments
    ///
    /// * `data_loader` - The source to load program data from. The Program will
    /// hold a pointer to this loader, which must outlive the returned Program
    /// instance.
    /// * `verification` - The type of verification to do before returning success.
    /// Defaults to `ProgramVerification::Minimal`.
    ///
    /// # Returns
    ///
    /// A new instance of Program.
    pub fn load(
        data_loader: &'a impl DataLoader,
        verification: Option<ProgramVerification>,
    ) -> Result<Self> {
        let mut data_loader = data_loader.data_loader();
        let verification = verification.unwrap_or(ProgramVerification::Minimal);
        Ok(Self(
            unsafe { et_c::Program::load(&mut *data_loader, verification) }.rs()?,
            PhantomData,
        ))
    }

    /// Returns the number of methods in the program.
    pub fn num_methods(&self) -> usize {
        unsafe { self.0.num_methods() }
    }

    /// Returns the name of the method at particular index.
    ///
    /// # Arguments
    ///
    /// * `method_index` - The index of the method name to retrieve. Must be less than the value returned by `num_methods()`.
    ///
    /// # Returns
    ///
    /// The name of the requested method. The pointer is owned by the Program, and has the same lifetime as the Program.
    pub fn get_method_name(&self, method_index: usize) -> Result<&'a str> {
        let method_name = unsafe { et_c::Program_get_method_name(&self.0, method_index) }.rs()?;
        Ok(unsafe { CStr::from_ptr(method_name).to_str().unwrap() })
    }

    /// Loads the named method and prepares it for execution.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to load.
    /// * `memory_manager` - The allocators to use during initialization and execution of the loaded method.
    ///
    /// # Returns
    ///
    /// The loaded method on success, or an error on failure.
    pub fn load_method(
        &self,
        method_name: &str,
        memory_manager: &'a MemoryManager,
    ) -> Result<Method> {
        let method_name = CString::new(method_name).unwrap();
        let mut memory_manager = memory_manager.0.borrow_mut();
        let event_tracer = ptr::null_mut(); // TODO: support event tracer
        let method = unsafe {
            self.0
                .load_method(method_name.as_ptr(), &mut *memory_manager, event_tracer)
        };
        Ok(Method(method.rs()?, PhantomData))
    }

    /// Gathers metadata for the named method.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to get metadata for.
    pub fn method_meta(&self, method_name: &str) -> Result<MethodMeta<'a>> {
        let method_name = CString::new(method_name).unwrap();
        let meta = unsafe { et_rs_c::Program_method_meta(&self.0, method_name.as_ptr()) }.rs()?;
        Ok(unsafe { MethodMeta::new(meta) })
    }

    /// Looks for an ExecuTorch program header in the provided data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data from the beginning of a file that might contain an ExecuTorch program.
    ///
    /// # Returns
    ///
    /// A value describing the presence of a header in the data.
    pub fn check_header(data: &[u8]) -> HeaderStatus {
        unsafe { et_c::Program::check_header(data.as_ptr() as *const _, data.len()) }
    }
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe { et_rs_c::Program_destructor(&mut self.0) };
    }
}

pub type ProgramVerification = et_c::Program_Verification;
pub type HeaderStatus = et_c::Program_HeaderStatus;

pub struct MethodMeta<'a>(et_c::MethodMeta, PhantomData<&'a ()>);
impl<'a> MethodMeta<'a> {
    pub(crate) unsafe fn new(meta: et_c::MethodMeta) -> Self {
        Self(meta, PhantomData)
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

pub struct Method<'a>(et_c::Method, PhantomData<&'a ()>);
impl<'a> Method<'a> {
    pub fn start_execution<'b>(&'b mut self) -> Execution<'a, 'b> {
        Execution::new(self)
    }

    pub fn inputs_size(&self) -> usize {
        unsafe { self.0.inputs_size() }
    }
}
impl Drop for Method<'_> {
    fn drop(&mut self) {
        unsafe { et_c::Method_Method_destructor(&mut self.0) };
    }
}

pub struct Execution<'a, 'b> {
    method: &'b mut Method<'a>,
    set_inputs: u64,
}
impl<'a, 'b> Execution<'a, 'b> {
    fn new(method: &'b mut Method<'a>) -> Self {
        assert!(
            method.inputs_size() <= u64::BITS as usize,
            "more that 64 inputs for method, unsupported"
        );
        Self {
            method,
            set_inputs: 0,
        }
    }

    pub fn set_input(&mut self, input: &'b EValue, input_idx: usize) -> Result<()> {
        unsafe { self.method.0.set_input(&input.0, input_idx) }.rs()?;
        self.set_inputs |= 1 << input_idx;
        Ok(())
    }

    pub fn execute(self) -> Result<Outputs<'a, 'b>> {
        assert_eq!(
            self.set_inputs,
            (1 << self.method.inputs_size()) - 1,
            "some inputs were not set"
        );
        unsafe { self.method.0.execute() }.rs()?;
        Ok(Outputs::new(self.method))
    }
}
pub struct Outputs<'a, 'b> {
    method: &'b mut Method<'a>,
}
impl<'a, 'b> Outputs<'a, 'b> {
    fn new(method: &'b mut Method<'a>) -> Self {
        Self { method }
    }

    pub fn get_output(&self, output_idx: usize) -> &'a EValue<'a> {
        let val = unsafe { &*self.method.0.get_output(output_idx) };
        // SAFETY: et_c::EValue as EValue has the same memory layout
        unsafe { std::mem::transmute::<&et_c::EValue, &EValue<'a>>(val) }
    }
}
