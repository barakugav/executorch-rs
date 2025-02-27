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
use crate::util::{ArrayRef, IntoCpp, IntoRust};
use crate::Result;
use executorch_sys as et_c;

/// A deserialized ExecuTorch program binary.
///
/// See the `examples/no_std` example for how to load and execute a program.
pub struct Program<'a>(et_c::Program, PhantomData<&'a ()>);
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
        let verification = verification.unwrap_or(ProgramVerification::Minimal).cpp();
        let program = try_new(|program| unsafe {
            et_c::executorch_Program_load(data_loader, verification, program)
        })?;
        Ok(Self(program, PhantomData))
    }

    /// Returns the number of methods in the program.
    pub fn num_methods(&self) -> usize {
        unsafe { et_c::executorch_Program_num_methods(&self.0) }
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
            et_c::executorch_Program_get_method_name(&self.0, method_index, method_name)
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
            et_c::executorch_Program_load_method(
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
            et_c::executorch_Program_method_meta(&self.0, method_name.as_ptr(), meta)
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
        unsafe { et_c::executorch_Program_check_header(data.as_ptr() as *const _, data.len()) }.rs()
    }
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe { et_c::executorch_Program_destructor(&mut self.0) };
    }
}

#[repr(u32)]
#[doc = " Types of validation that the Program can do before parsing the data."]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ProgramVerification {
    #[doc = " Do minimal verification of the data, ensuring that the header appears\n correct.\n\n Has minimal runtime overhead."]
    Minimal = 0,
    #[doc = " Do full verification of the data, ensuring that internal pointers are\n self-consistent and that the data has not been truncated or obviously\n corrupted. May not catch all types of corruption, but should guard\n against illegal memory operations during parsing.\n\n Will have higher runtime overhead, scaling with the complexity of the\n proram data."]
    InternalConsistency = 1,
}
impl IntoCpp for ProgramVerification {
    type CppType = et_c::ProgramVerification;
    fn cpp(self) -> Self::CppType {
        match self {
            ProgramVerification::Minimal => et_c::ProgramVerification::ProgramVerification_Minimal,
            ProgramVerification::InternalConsistency => {
                et_c::ProgramVerification::ProgramVerification_InternalConsistency
            }
        }
    }
}

#[repr(u32)]
#[doc = " Describes the presence of an ExecuTorch program header."]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum HeaderStatus {
    #[doc = " An ExecuTorch program header is present, and its version is compatible\n with this version of the runtime."]
    CompatibleVersion = 0,
    #[doc = " An ExecuTorch program header is present, but its version is not\n compatible with this version of the runtime."]
    IncompatibleVersion = 1,
    #[doc = " An ExecuTorch program header is not present."]
    NotPresent = 2,
    #[doc = " The data provided was too short to find the program header."]
    ShortData = 3,
}
impl IntoRust for et_c::ProgramHeaderStatus {
    type RsType = HeaderStatus;
    fn rs(self) -> Self::RsType {
        match self {
            et_c::ProgramHeaderStatus::ProgramHeaderStatus_CompatibleVersion => {
                HeaderStatus::CompatibleVersion
            }
            et_c::ProgramHeaderStatus::ProgramHeaderStatus_IncompatibleVersion => {
                HeaderStatus::IncompatibleVersion
            }
            et_c::ProgramHeaderStatus::ProgramHeaderStatus_NotPresent => HeaderStatus::NotPresent,
            et_c::ProgramHeaderStatus::ProgramHeaderStatus_ShortData => HeaderStatus::ShortData,
        }
    }
}

/// Describes a a method in an ExecuTorch program.
///
/// The program used to create a MethodMeta object must outlive the MethodMeta.
/// It is separate from Method so that this information can be accessed without
/// paying the initialization cost of loading the full Method.
pub struct MethodMeta<'a>(et_c::MethodMeta, PhantomData<&'a ()>);
impl MethodMeta<'_> {
    pub(crate) unsafe fn new(meta: et_c::MethodMeta) -> Self {
        Self(meta, PhantomData)
    }

    /// Get the name of this method.
    pub fn name(&self) -> &str {
        let name = unsafe { et_c::executorch_MethodMeta_name(&self.0) };
        unsafe { CStr::from_ptr(name).to_str().unwrap() }
    }

    /// Get the number of inputs to this method.
    pub fn num_inputs(&self) -> usize {
        unsafe { et_c::executorch_MethodMeta_num_inputs(&self.0) }
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
        try_new(|tag| unsafe { et_c::executorch_MethodMeta_input_tag(&self.0, idx, tag) })
            .map(IntoRust::rs)
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
        let info = try_new(|info| unsafe {
            et_c::executorch_MethodMeta_input_tensor_meta(&self.0, idx, info)
        })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of outputs to this method.
    pub fn num_outputs(&self) -> usize {
        unsafe { et_c::executorch_MethodMeta_num_outputs(&self.0) }
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
        try_new(|tag| unsafe { et_c::executorch_MethodMeta_output_tag(&self.0, idx, tag) })
            .map(IntoRust::rs)
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
        let info = try_new(|info| unsafe {
            et_c::executorch_MethodMeta_output_tensor_meta(&self.0, idx, info)
        })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of memory-planned buffers this method requires.
    pub fn num_memory_planned_buffers(&self) -> usize {
        unsafe { et_c::executorch_MethodMeta_num_memory_planned_buffers(&self.0) }
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
            et_c::executorch_MethodMeta_memory_planned_buffer_size(&self.0, idx, size)
        })?;
        Ok(size as usize)
    }
}

/// Metadata about a specific tensor of an ExecuTorch Program.
///
/// The program used to create the MethodMeta object that created this
/// TensorInfo must outlive this TensorInfo.
pub struct TensorInfo<'a>(et_c::TensorInfo, PhantomData<&'a ()>);
impl<'a> TensorInfo<'a> {
    pub(crate) unsafe fn new(info: et_c::TensorInfo) -> Self {
        Self(info, PhantomData)
    }

    /// Returns the sizes of the tensor.
    pub fn sizes(&self) -> &'a [i32] {
        let span = unsafe { et_c::executorch_TensorInfo_sizes(&self.0) };
        unsafe { ArrayRef::from_inner(span) }.as_slice()
    }

    /// Returns the dim order of the tensor.
    pub fn dim_order(&self) -> &'a [u8] {
        let span = unsafe { et_c::executorch_TensorInfo_dim_order(&self.0) };
        unsafe { ArrayRef::from_inner(span) }.as_slice()
    }

    /// Returns the scalar type of the input/output.
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { et_c::executorch_TensorInfo_scalar_type(&self.0) }.rs()
    }

    /// Returns the size of the tensor in bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { et_c::executorch_TensorInfo_nbytes(&self.0) }
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
pub struct Method<'a>(et_c::Method, PhantomData<&'a ()>);
impl Method<'_> {
    /// Starts the execution of the method.
    pub fn start_execution(&mut self) -> Execution {
        Execution::new(&mut self.0)
    }

    /// Returns the number of inputs the Method expects.
    pub fn inputs_size(&self) -> usize {
        unsafe { et_c::executorch_Method_inputs_size(&self.0) }
    }
}
impl Drop for Method<'_> {
    fn drop(&mut self) {
        unsafe { et_c::executorch_Method_destructor(&mut self.0) };
    }
}

/// An method execution builder used to set inputs and execute the method.
pub struct Execution<'a> {
    method: &'a mut et_c::Method,
    set_inputs: u64,
}
impl<'a> Execution<'a> {
    fn new(method: &'a mut et_c::Method) -> Self {
        assert!(
            unsafe { et_c::executorch_Method_inputs_size(method) } <= u64::BITS as usize,
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
        unsafe {
            et_c::executorch_Method_set_input(self.method, input.as_evalue() as *const _, input_idx)
        }
        .rs()?;
        self.set_inputs |= 1 << input_idx;
        Ok(())
    }

    /// Execute the method.
    pub fn execute(self) -> Result<Outputs<'a>> {
        assert_eq!(
            self.set_inputs,
            (1 << unsafe { et_c::executorch_Method_inputs_size(self.method) }) - 1,
            "some inputs were not set"
        );
        unsafe { et_c::executorch_Method_execute(self.method) }.rs()?;
        Ok(Outputs::new(self.method))
    }
}

/// The outputs of a method execution.
///
/// Access the outputs of a method execution by indexing into the Outputs object.
pub struct Outputs<'a> {
    method: &'a mut et_c::Method,
}
impl<'a> Outputs<'a> {
    fn new(method: &'a mut et_c::Method) -> Self {
        Self { method }
    }

    /// Returns the number of outputs the Method returns.
    pub fn len(&self) -> usize {
        unsafe { et_c::executorch_Method_outputs_size(self.method) }
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
        let value = unsafe { &*et_c::executorch_Method_get_output(self.method as *const _, index) };
        EValue::from_inner_ref(value)
    }
}
