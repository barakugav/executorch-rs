//! Lower-level API for loading and executing ExecuTorch programs.
//!
//! This module is the lowest level API for the ExecuTorch library. It provides the ability to load and execute
//! programs, while controlling memory allocation and execution.
//!
//! See the `examples/no_std` example for how to load and execute a program.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::ptr;

use crate::data_loader::DataLoader;
use crate::error::try_new;
use crate::evalue::{EValue, Tag};
use crate::memory::MemoryManager;
use crate::tensor::ScalarType;
use crate::util::{ArrayRef, IntoRust};
use crate::{et_c, et_rs_c, Result};

/// A deserialized ExecuTorch program binary.
///
/// See the `examples/no_std` example for how to load and execute a program.
pub struct Program<'a>(et_c::runtime::Program, PhantomData<&'a ()>);
impl<'a> Program<'a> {
    /// Loads a Program from the provided loader. The Program will hold a pointer
    /// to the loader, which must outlive the returned Program instance.
    ///
    /// # Arguments
    ///
    /// * `data_loader` - The source to load program data from. The Program will
    ///     hold a pointer to this loader, which must outlive the returned Program
    ///     instance.
    /// * `verification` - The type of verification to do before returning success.
    ///     Defaults to `ProgramVerification::Minimal`.
    ///
    /// # Returns
    ///
    /// A new instance of Program.
    pub fn load(
        data_loader: &'a impl AsRef<DataLoader>,
        verification: Option<ProgramVerification>,
    ) -> Result<Self> {
        let data_loader = data_loader.as_ref().0.get();
        let verification = verification.unwrap_or(ProgramVerification::Minimal);
        let program = try_new(|program| unsafe {
            et_rs_c::Program_load(data_loader, verification, program)
        })?;
        Ok(Self(program, PhantomData))
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
    pub fn get_method_name(&self, method_index: usize) -> Result<&str> {
        let method_name = try_new(|method_name| unsafe {
            et_rs_c::Program_get_method_name(&self.0, method_index, method_name)
        })?;
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
    pub fn load_method<'b>(
        &'b self,
        method_name: &CStr,
        memory_manager: &'b MemoryManager,
    ) -> Result<Method<'b>> {
        let memory_manager = memory_manager.0.get();
        let event_tracer = ptr::null_mut(); // TODO: support event tracer
        let method = try_new(|method| unsafe {
            et_rs_c::Program_load_method(
                &self.0,
                method_name.as_ptr(),
                memory_manager,
                event_tracer,
                method,
            )
        })?;
        Ok(Method(method, PhantomData))
    }

    /// Gathers metadata for the named method.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to get metadata for.
    pub fn method_meta(&self, method_name: &CStr) -> Result<MethodMeta> {
        let meta = try_new(|meta| unsafe {
            et_rs_c::Program_method_meta(&self.0, method_name.as_ptr(), meta)
        })?;
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
        unsafe { et_c::runtime::Program::check_header(data.as_ptr() as *const _, data.len()) }
    }
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe { et_rs_c::Program_destructor(&mut self.0) };
    }
}

/// Types of validation that the Program can do before parsing the data.
pub type ProgramVerification = et_c::runtime::Program_Verification;
/// Describes the presence of an ExecuTorch program header.
pub type HeaderStatus = et_c::runtime::Program_HeaderStatus;

/// Describes a a method in an ExecuTorch program.
///
/// The program used to create a MethodMeta object must outlive the MethodMeta.
/// It is separate from Method so that this information can be accessed without
/// paying the initialization cost of loading the full Method.
pub struct MethodMeta<'a>(et_c::runtime::MethodMeta, PhantomData<&'a ()>);
impl MethodMeta<'_> {
    pub(crate) unsafe fn new(meta: et_c::runtime::MethodMeta) -> Self {
        Self(meta, PhantomData)
    }

    /// Get the name of this method.
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.0.name()).to_str().unwrap() }
    }

    /// Get the number of inputs to this method.
    pub fn num_inputs(&self) -> usize {
        unsafe { self.0.num_inputs() }
    }

    /// Get the tag of the specified input.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the input to look up.
    ///
    /// # Returns
    ///
    /// The tag of input, can only be [Tensor, Int, Bool, Double, String].
    pub fn input_tag(&self, idx: usize) -> Result<Tag> {
        try_new(|tag| unsafe { et_rs_c::MethodMeta_input_tag(&self.0, idx, tag) })
    }

    /// Get metadata about the specified input.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the input to look up.
    ///
    /// # Returns
    ///
    /// The metadata on success, or an error on failure. Only valid for `Tag::Tensor`
    pub fn input_tensor_meta(&self, idx: usize) -> Result<TensorInfo> {
        let info =
            try_new(|info| unsafe { et_rs_c::MethodMeta_input_tensor_meta(&self.0, idx, info) })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of outputs to this method.
    pub fn num_outputs(&self) -> usize {
        unsafe { self.0.num_outputs() }
    }

    /// Get the tag of the specified output.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the output to look up.
    ///
    /// # Returns
    ///
    /// The tag of output, can only be [Tensor, Int, Bool, Double, String].
    pub fn output_tag(&self, idx: usize) -> Result<Tag> {
        try_new(|tag| unsafe { et_rs_c::MethodMeta_output_tag(&self.0, idx, tag) })
    }

    /// Get metadata about the specified output.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the output to look up.
    ///
    /// # Returns
    ///
    /// The metadata on success, or an error on failure. Only valid for `Tag::Tensor`
    pub fn output_tensor_meta(&self, idx: usize) -> Result<TensorInfo> {
        let info =
            try_new(|info| unsafe { et_rs_c::MethodMeta_output_tensor_meta(&self.0, idx, info) })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of memory-planned buffers this method requires.
    pub fn num_memory_planned_buffers(&self) -> usize {
        unsafe { self.0.num_memory_planned_buffers() }
    }

    /// Get the size in bytes of the specified memory-planned buffer.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the buffer to look up.
    ///
    /// # Returns
    ///
    /// The size in bytes on success, or an error on failure.
    pub fn memory_planned_buffer_size(&self, idx: usize) -> Result<usize> {
        let size = try_new(|size| unsafe {
            et_rs_c::MethodMeta_memory_planned_buffer_size(&self.0, idx, size)
        })?;
        Ok(size as usize)
    }
}

/// Metadata about a specific tensor of an ExecuTorch Program.
///
/// The program used to create the MethodMeta object that created this
/// TensorInfo must outlive this TensorInfo.
pub struct TensorInfo<'a>(et_c::runtime::TensorInfo, PhantomData<&'a ()>);
impl<'a> TensorInfo<'a> {
    pub(crate) unsafe fn new(info: et_c::runtime::TensorInfo) -> Self {
        Self(info, PhantomData)
    }

    /// Returns the sizes of the tensor.
    pub fn sizes(&self) -> &'a [i32] {
        let span = unsafe { et_rs_c::TensorInfo_sizes(&self.0) };
        unsafe { ArrayRef::from_inner(span) }.as_slice()
    }

    /// Returns the dim order of the tensor.
    pub fn dim_order(&self) -> &'a [u8] {
        let span = unsafe { et_rs_c::TensorInfo_dim_order(&self.0) };
        unsafe { ArrayRef::from_inner(span) }.as_slice()
    }

    /// Returns the scalar type of the input/output.
    pub fn scalar_type(&self) -> Option<ScalarType> {
        let scalar_type = unsafe { et_c::runtime::TensorInfo_scalar_type(&self.0) };
        ScalarType::from_c_scalar_type(scalar_type)
    }

    /// Returns the size of the tensor in bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { et_c::runtime::TensorInfo_nbytes(&self.0) }
    }
}
impl std::fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("TensorInfo")
            .field("sizes", &self.sizes())
            .field("dim_order", &self.dim_order())
            .field("scalar_type", &self.scalar_type())
            .field("nbytes", &self.nbytes())
            .finish()
    }
}

/// An executable method of an ExecuTorch program. Maps to a python method like
/// `forward()` on the original `nn.Module`.
pub struct Method<'a>(et_c::runtime::Method, PhantomData<&'a ()>);
impl Method<'_> {
    /// Starts the execution of the method.
    pub fn start_execution(&mut self) -> Execution {
        Execution::new(&mut self.0)
    }

    /// Returns the number of inputs the Method expects.
    pub fn inputs_size(&self) -> usize {
        unsafe { self.0.inputs_size() }
    }
}
impl Drop for Method<'_> {
    fn drop(&mut self) {
        unsafe { et_c::runtime::Method_Method_destructor(&mut self.0) };
    }
}

/// An method execution builder used to set inputs and execute the method.
pub struct Execution<'a> {
    method: &'a mut et_c::runtime::Method,
    set_inputs: u64,
}
impl<'a> Execution<'a> {
    fn new(method: &'a mut et_c::runtime::Method) -> Self {
        assert!(
            unsafe { method.inputs_size() } <= u64::BITS as usize,
            "more that 64 inputs for method, unsupported"
        );
        Self {
            method,
            set_inputs: 0,
        }
    }

    /// Sets the internal input value to be equivalent to the provided value.
    ///
    /// # Arguments
    ///
    /// * `input` - The evalue to copy into the method input. If the evalue is a tensor, the data is copied in most
    ///     cases, so the tensor passed in here does not always need to outlive this call. But there is a case where the
    ///     Method will keep a pointer to the tensor's data. Based on the memory plan of the method, the inputs may not
    ///     have buffer space pre-allocated for them. In this case the executor will alias the memory of the tensors
    ///     provided as inputs here rather then deepcopy the input into the memory planned arena.
    /// * `input_idx` - Zero-based index of the input to set. Must be less than the value returned by inputs_size().
    pub fn set_input(&mut self, input: &'a EValue, input_idx: usize) -> Result<()> {
        unsafe { self.method.set_input(input.as_evalue(), input_idx) }.rs()?;
        self.set_inputs |= 1 << input_idx;
        Ok(())
    }

    /// Execute the method.
    pub fn execute(self) -> Result<Outputs<'a>> {
        assert_eq!(
            self.set_inputs,
            (1 << unsafe { self.method.inputs_size() }) - 1,
            "some inputs were not set"
        );
        unsafe { self.method.execute() }.rs()?;
        Ok(Outputs::new(self.method))
    }
}

/// The outputs of a method execution.
///
/// Access the outputs of a method execution by indexing into the Outputs object.
pub struct Outputs<'a> {
    method: &'a mut et_c::runtime::Method,
}
impl<'a> Outputs<'a> {
    fn new(method: &'a mut et_c::runtime::Method) -> Self {
        Self { method }
    }

    /// Returns the number of outputs the Method returns.
    pub fn len(&self) -> usize {
        unsafe { self.method.outputs_size() }
    }

    /// Returns true if the Method returns no outputs.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the output at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> EValue {
        let value = unsafe { &*self.method.get_output(index) };
        EValue::from_inner_ref(value)
    }
}
