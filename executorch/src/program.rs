//! Lower-level API for loading and executing ExecuTorch programs.
//!
//! This module is the lowest level API for the ExecuTorch library. It provides the ability to load and execute
//! programs, while controlling memory allocation and execution.
//!
//! ```rust,ignore
//! let mut buffer = [0_u8; 4096];
//! let allocator = BufferMemoryAllocator::new(&mut buffer);
//!
//! let data_loader = BufferDataLoader::new(ADD_MODEL_BYTES);
//! let program = Program::load(data_loader.as_ref(), None).unwrap();
//!
//! let method_meta = program.method_meta(c"forward").unwrap();
//!
//! let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
//! let planned_arenas = allocator
//!     .allocate_arr_fn(num_memory_planned_buffers, |idx| {
//!         let buf_size = method_meta.memory_planned_buffer_size(idx).unwrap();
//!         Span::from_slice(allocator.allocate_arr::<u8>(buf_size).unwrap())
//!     })
//!     .unwrap();
//!
//! let mut planned_memory = HierarchicalAllocator::new(planned_arenas);
//!
//! let memory_manager = MemoryManager::new(&allocator, Some(&mut planned_memory), None);
//!
//! let mut method = program
//!     .load_method(c"forward", &memory_manager, None, None)
//!     .unwrap();
//!
//! let input_array1 = ArrayStorage::new(array!(1.0_f32)).unwrap();
//! let input_tensor_impl1 = input_array1.as_tensor_impl();
//! let input_tensor1 = Tensor::new_in_allocator(&input_tensor_impl1, &allocator);
//! let input_evalue1 = EValue::new_in_allocator(input_tensor1, &allocator);
//!
//! let input_array2 = ArrayStorage::new(array!(1.0_f32)).unwrap();
//! let input_tensor_impl2 = input_array2.as_tensor_impl();
//! let input_tensor2 = Tensor::new_in_allocator(&input_tensor_impl2, &allocator);
//! let input_evalue2 = EValue::new_in_allocator(input_tensor2, &allocator);
//!
//! let mut method_exe = method.start_execution();
//!
//! method_exe.set_input(&input_evalue1, 0).unwrap();
//! method_exe.set_input(&input_evalue2, 1).unwrap();
//!
//! let outputs = method_exe.execute().unwrap();
//! let output = outputs.get(0);
//! let output = output.as_tensor().into_typed::<f32>();
//!
//! assert_eq!(array!(2.0), output.as_array());
//! ```
//!
//! See the `examples/no_std` example for how to load and execute a program.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::ptr;

use crate::data_loader::DataLoader;
use crate::data_map::NamedDataMap;
use crate::evalue::{EValue, Tag};
use crate::event_tracer::EventTracer;
use crate::memory::MemoryManager;
use crate::tensor::ScalarType;
use crate::util::{try_c_new, ArrayRef, IntoCpp, IntoRust, __ArrayRefImpl, chars2str, FfiChar};
use crate::{sys, Error, Result};

/// A deserialized ExecuTorch program binary.
///
/// See the `examples/no_std` example for how to load and execute a program.
pub struct Program<'a>(sys::Program, PhantomData<&'a ()>);
impl<'a> Program<'a> {
    /// Loads a Program from the provided loader. The Program will hold a pointer
    /// to the loader, which must outlive the returned Program instance.
    ///
    /// # Arguments
    ///
    /// * `data_loader` - The source to load program data from. The Program will
    ///   hold a pointer to this loader, which must outlive the returned Program
    ///   instance.
    /// * `verification` - The type of verification to do before returning success.
    ///   Defaults to `ProgramVerification::Minimal`.
    ///
    /// # Returns
    ///
    /// A new instance of Program.
    pub fn load(
        data_loader: &'a DataLoader,
        verification: Option<ProgramVerification>,
    ) -> Result<Self> {
        let data_loader = sys::DataLoaderRefMut {
            ptr: data_loader as *const _ as *mut _,
        };
        let verification = verification.unwrap_or(ProgramVerification::Minimal).cpp();
        let program = try_c_new(|program| unsafe {
            sys::executorch_Program_load(data_loader, verification, program)
        })?;
        Ok(Self(program, PhantomData))
    }

    /// Returns the number of methods in the program.
    pub fn num_methods(&self) -> usize {
        unsafe { sys::executorch_Program_num_methods(&self.0) }
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
        let method_name = try_c_new(|method_name| unsafe {
            sys::executorch_Program_get_method_name(&self.0, method_index, method_name)
        })?;
        let method_name = unsafe { CStr::from_ptr(method_name) };
        method_name.to_str().map_err(|_| Error::InvalidString)
    }

    /// Get the named data map from the program.
    pub fn get_named_data_map(&self) -> Result<&NamedDataMap> {
        let data_map = try_c_new(|data_map| unsafe {
            sys::executorch_Program_get_named_data_map(&self.0, data_map)
        })?;
        Ok(unsafe { &*data_map.ptr.cast() })
    }

    /// Loads the named method and prepares it for execution.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to load.
    /// * `memory_manager` - The allocators to use during initialization and execution of the loaded method.
    /// * `event_tracer` - The event tracer to use for this method run.
    /// * `named_data_map` - An optional map of {name, blob} used to resolve data that is external to the PTE, if any.
    ///
    /// # Returns
    ///
    /// The loaded method on success, or an error on failure.
    pub fn load_method<'b>(
        &'b self,
        method_name: &CStr,
        memory_manager: &'b MemoryManager,
        event_tracer: Option<&'b mut EventTracer>,
        named_data_map: Option<&'b NamedDataMap>,
    ) -> Result<Method<'b>> {
        let memory_manager = memory_manager.0.get();
        let event_tracer = event_tracer
            .map(|tracer| tracer as *mut EventTracer)
            .unwrap_or(ptr::null_mut());
        let event_tracer = sys::EventTracerRefMut {
            ptr: event_tracer as *mut _,
        };
        let named_data_map = named_data_map
            .map(|map| map as *const _ as *const _)
            .unwrap_or(ptr::null());
        let named_data_map = sys::NamedDataMapRef {
            ptr: named_data_map,
        };
        let method = try_c_new(|method| unsafe {
            sys::executorch_Program_load_method(
                &self.0,
                method_name.as_ptr(),
                memory_manager,
                event_tracer,
                named_data_map,
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
    pub fn method_meta(&self, method_name: &CStr) -> Result<MethodMeta<'_>> {
        let meta = try_c_new(|meta| unsafe {
            sys::executorch_Program_method_meta(&self.0, method_name.as_ptr(), meta)
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
        unsafe { sys::executorch_Program_check_header(data.as_ptr() as *const _, data.len()) }.rs()
    }
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe { sys::executorch_Program_destructor(&mut self.0) };
    }
}

#[repr(u8)]
#[doc = " Types of validation that the Program can do before parsing the data."]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ProgramVerification {
    #[doc = " Do minimal verification of the data, ensuring that the header appears\n correct.\n\n Has minimal runtime overhead."]
    Minimal = sys::ProgramVerification::ProgramVerification_Minimal as u8,
    #[doc = " Do full verification of the data, ensuring that internal pointers are\n self-consistent and that the data has not been truncated or obviously\n corrupted. May not catch all types of corruption, but should guard\n against illegal memory operations during parsing.\n\n Will have higher runtime overhead, scaling with the complexity of the\n proram data."]
    InternalConsistency = sys::ProgramVerification::ProgramVerification_InternalConsistency as u8,
}
impl IntoCpp for ProgramVerification {
    type CppType = sys::ProgramVerification;
    fn cpp(self) -> Self::CppType {
        match self {
            ProgramVerification::Minimal => sys::ProgramVerification::ProgramVerification_Minimal,
            ProgramVerification::InternalConsistency => {
                sys::ProgramVerification::ProgramVerification_InternalConsistency
            }
        }
    }
}

#[repr(u32)]
#[doc = " Describes the presence of an ExecuTorch program header."]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum HeaderStatus {
    #[doc = " An ExecuTorch program header is present, and its version is compatible\n with this version of the runtime."]
    CompatibleVersion = sys::ProgramHeaderStatus::ProgramHeaderStatus_CompatibleVersion as u32,
    #[doc = " An ExecuTorch program header is present, but its version is not\n compatible with this version of the runtime."]
    IncompatibleVersion = sys::ProgramHeaderStatus::ProgramHeaderStatus_IncompatibleVersion as u32,
    #[doc = " An ExecuTorch program header is not present."]
    NotPresent = sys::ProgramHeaderStatus::ProgramHeaderStatus_NotPresent as u32,
    #[doc = " The data provided was too short to find the program header."]
    ShortData = sys::ProgramHeaderStatus::ProgramHeaderStatus_ShortData as u32,
}
impl IntoRust for sys::ProgramHeaderStatus {
    type RsType = HeaderStatus;
    fn rs(self) -> Self::RsType {
        match self {
            sys::ProgramHeaderStatus::ProgramHeaderStatus_CompatibleVersion => {
                HeaderStatus::CompatibleVersion
            }
            sys::ProgramHeaderStatus::ProgramHeaderStatus_IncompatibleVersion => {
                HeaderStatus::IncompatibleVersion
            }
            sys::ProgramHeaderStatus::ProgramHeaderStatus_NotPresent => HeaderStatus::NotPresent,
            sys::ProgramHeaderStatus::ProgramHeaderStatus_ShortData => HeaderStatus::ShortData,
        }
    }
}

/// Describes a a method in an ExecuTorch program.
///
/// The program used to create a MethodMeta object must outlive the MethodMeta.
/// It is separate from Method so that this information can be accessed without
/// paying the initialization cost of loading the full Method.
pub struct MethodMeta<'a>(sys::MethodMeta, PhantomData<&'a ()>);
impl MethodMeta<'_> {
    pub(crate) unsafe fn new(meta: sys::MethodMeta) -> Self {
        Self(meta, PhantomData)
    }

    /// Get the name of this method.
    pub fn name(&self) -> &str {
        let name = unsafe { sys::executorch_MethodMeta_name(&self.0) };
        let name = unsafe { CStr::from_ptr(name) };
        name.to_str().map_err(|_| Error::InvalidString).unwrap()
    }

    /// Get the number of inputs to this method.
    pub fn num_inputs(&self) -> usize {
        unsafe { sys::executorch_MethodMeta_num_inputs(&self.0) }
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
        try_c_new(|tag| unsafe { sys::executorch_MethodMeta_input_tag(&self.0, idx, tag) })
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
    pub fn input_tensor_meta(&self, idx: usize) -> Result<TensorInfo<'_>> {
        let info = try_c_new(|info| unsafe {
            sys::executorch_MethodMeta_input_tensor_meta(&self.0, idx, info)
        })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of outputs to this method.
    pub fn num_outputs(&self) -> usize {
        unsafe { sys::executorch_MethodMeta_num_outputs(&self.0) }
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
        try_c_new(|tag| unsafe { sys::executorch_MethodMeta_output_tag(&self.0, idx, tag) })
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
    pub fn output_tensor_meta(&self, idx: usize) -> Result<TensorInfo<'_>> {
        let info = try_c_new(|info| unsafe {
            sys::executorch_MethodMeta_output_tensor_meta(&self.0, idx, info)
        })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of attribute tensors in this method.
    pub fn num_attributes(&self) -> usize {
        unsafe { sys::executorch_MethodMeta_num_attributes(&self.0) }
    }

    /// Get metadata about the specified attribute tensor.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the attribute tensor to look up. Must be in range `0..num_attributes()`.
    ///
    /// # Returns
    ///
    /// The metadata on success, or an error on failure.
    pub fn attribute_tensor_meta(&self, idx: usize) -> Result<TensorInfo<'_>> {
        let info = try_c_new(|info| unsafe {
            sys::executorch_MethodMeta_attribute_tensor_meta(&self.0, idx, info)
        })?;
        Ok(unsafe { TensorInfo::new(info) })
    }

    /// Get the number of memory-planned buffers this method requires.
    pub fn num_memory_planned_buffers(&self) -> usize {
        unsafe { sys::executorch_MethodMeta_num_memory_planned_buffers(&self.0) }
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
        let size = try_c_new(|size| unsafe {
            sys::executorch_MethodMeta_memory_planned_buffer_size(&self.0, idx, size)
        })?;
        Ok(size as usize)
    }

    /// Check to see if a backend is used in this method.
    pub fn uses_backend(&self, backend_name: &CStr) -> bool {
        unsafe { sys::executorch_MethodMeta_uses_backend(&self.0, backend_name.as_ptr()) }
    }

    /// Get the number of backends used in this method.
    pub fn num_backends(&self) -> usize {
        unsafe { sys::executorch_MethodMeta_num_backends(&self.0) }
    }

    /// Get the backend name at the given index.
    pub fn get_backend_name(&self, index: usize) -> Result<&str> {
        let backend_name = try_c_new(|name| unsafe {
            sys::executorch_MethodMeta_get_backend_name(&self.0, index, name)
        })?;
        let backend_name = unsafe { CStr::from_ptr(backend_name) };
        backend_name.to_str().map_err(|_| Error::InvalidString)
    }
}

/// Metadata about a specific tensor of an ExecuTorch Program.
///
/// The program used to create the MethodMeta object that created this
/// TensorInfo must outlive this TensorInfo.
pub struct TensorInfo<'a>(sys::TensorInfo, PhantomData<&'a ()>);
impl<'a> TensorInfo<'a> {
    pub(crate) unsafe fn new(info: sys::TensorInfo) -> Self {
        Self(info, PhantomData)
    }

    /// Returns the sizes of the tensor.
    pub fn sizes(&self) -> &[i32] {
        let span = unsafe { sys::executorch_TensorInfo_sizes(&self.0) };
        unsafe { ArrayRef::from_inner(span) }.as_slice()
    }

    /// Returns the dim order of the tensor.
    pub fn dim_order(&self) -> &[u8] {
        let span = unsafe { sys::executorch_TensorInfo_dim_order(&self.0) };
        unsafe { ArrayRef::from_inner(span) }.as_slice()
    }

    /// Returns the scalar type of the input/output.
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { sys::executorch_TensorInfo_scalar_type(&self.0) }.rs()
    }

    /// Returns whether the tensor's memory was planned during export.
    pub fn is_memory_planned(&self) -> bool {
        unsafe { sys::executorch_TensorInfo_is_memory_planned(&self.0) }
    }

    /// Returns the size of the tensor in bytes.
    pub fn nbytes(&self) -> usize {
        unsafe { sys::executorch_TensorInfo_nbytes(&self.0) }
    }

    /// Returns the fully qualified name of the Tensor.
    ///
    /// Might be empty if the tensor is nameless.
    ///
    /// The function calls [`Self::name_chars`] internally, and tries to convert the returned bytes to a `&str`.
    pub fn name(&self) -> Result<&str, std::str::Utf8Error> {
        chars2str(self.name_chars())
    }

    /// Returns the fully qualified name of the Tensor as `[ffi::c_char]`.
    ///
    /// Might be empty if the tensor is nameless.
    pub fn name_chars(&self) -> &[std::ffi::c_char] {
        let chars = unsafe { sys::executorch_TensorInfo_name(&self.0).as_slice() };
        unsafe { std::mem::transmute::<&[FfiChar], &[std::ffi::c_char]>(chars) }
    }
}
impl std::fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("TensorInfo")
            .field("name", &self.name().unwrap_or("<invalid utf8>"))
            .field("sizes", &self.sizes())
            .field("dim_order", &self.dim_order())
            .field("scalar_type", &self.scalar_type())
            .field("nbytes", &self.nbytes())
            .finish()
    }
}

/// An executable method of an ExecuTorch program. Maps to a python method like
/// `forward()` on the original `nn.Module`.
pub struct Method<'a>(sys::Method, PhantomData<&'a ()>);
impl Method<'_> {
    /// Starts the execution of the method.
    pub fn start_execution(&mut self) -> Execution<'_> {
        Execution::new(&mut self.0)
    }

    /// Returns the number of inputs the Method expects.
    pub fn inputs_size(&self) -> usize {
        unsafe { sys::executorch_Method_inputs_size(&self.0) }
    }

    /// Retrieves the attribute tensor associated with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the attribute tensor to retrieve.
    ///
    /// # Returns
    ///
    /// Result containing the attribute tensor on success, non-Ok on failure.
    #[cfg(feature = "alloc")]
    pub fn get_attribute<'b>(&'b mut self, name: &str) -> Result<crate::tensor::TensorAny<'b>> {
        let name = crate::util::str2chars(name);
        let name = unsafe { std::mem::transmute::<&[std::ffi::c_char], &[FfiChar]>(name) };
        let name = ArrayRef::from_slice(name);

        // Safety: sys::executorch_Method_get_attribute writes to the tensor pointer.
        let tensor = unsafe {
            crate::util::NonTriviallyMovable::try_new_boxed(|tensor: *mut sys::TensorStorage| {
                let tensor = sys::TensorRefMut { ptr: tensor.cast() };
                sys::executorch_Method_get_attribute(&mut self.0, name.0, tensor).rs()
            })?
        };

        // Safety: The created tensor is immutable, therefore there is no risk for UB
        unsafe {
            Ok(crate::tensor::TensorAny::from_raw_tensor(
                crate::tensor::RawTensor::new_impl(tensor),
            ))
        }
    }
}
impl Drop for Method<'_> {
    fn drop(&mut self) {
        unsafe { sys::executorch_Method_destructor(&mut self.0) };
    }
}

/// An method execution builder used to set inputs and execute the method.
pub struct Execution<'a> {
    method: &'a mut sys::Method,
    set_inputs: u64,
}
impl<'a> Execution<'a> {
    fn new(method: &'a mut sys::Method) -> Self {
        assert!(
            unsafe { sys::executorch_Method_inputs_size(method) } <= u64::BITS as usize,
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
    ///   cases, so the tensor passed in here does not always need to outlive this call. But there is a case where the
    ///   Method will keep a pointer to the tensor's data. Based on the memory plan of the method, the inputs may not
    ///   have buffer space pre-allocated for them. In this case the executor will alias the memory of the tensors
    ///   provided as inputs here rather then deepcopy the input into the memory planned arena.
    /// * `input_idx` - Zero-based index of the input to set. Must be less than the value returned by inputs_size().
    pub fn set_input(&mut self, input: &'a EValue, input_idx: usize) -> Result<()> {
        unsafe { sys::executorch_Method_set_input(self.method, input.cpp(), input_idx) }.rs()?;
        self.set_inputs |= 1 << input_idx;
        Ok(())
    }

    /// Execute the method.
    ///
    /// # Errors
    ///
    /// Returns an error if not all inputs were set before executing the method, or if the execution failed.
    pub fn execute(self) -> Result<Outputs<'a>> {
        if self.set_inputs != (1 << unsafe { sys::executorch_Method_inputs_size(self.method) }) - 1
        {
            crate::log::error!("Not all inputs were set before executing the method");
            return Err(Error::InvalidArgument);
        }
        unsafe { sys::executorch_Method_execute(self.method) }.rs()?;
        Ok(Outputs::new(self.method))
    }
}

/// The outputs of a method execution.
///
/// Access the outputs of a method execution by indexing into the Outputs object.
pub struct Outputs<'a> {
    method: &'a mut sys::Method,
}
impl<'a> Outputs<'a> {
    fn new(method: &'a mut sys::Method) -> Self {
        Self { method }
    }

    /// Returns the number of outputs the Method returns.
    pub fn len(&self) -> usize {
        unsafe { sys::executorch_Method_outputs_size(self.method) }
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
    pub fn get(&self, index: usize) -> EValue<'_> {
        let value = unsafe { sys::executorch_Method_get_output(self.method as *const _, index) };
        unsafe { EValue::from_inner_ref(value) }
    }
}

#[cfg(test)]
mod tests {
    use crate::data_loader::BufferDataLoader;
    use crate::tests::ADD_MODEL_BYTES;

    use super::*;

    #[test]
    fn load() {
        let loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(loader.as_ref(), None);
        assert!(program.is_ok());

        let loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(loader.as_ref(), Some(ProgramVerification::Minimal));
        assert!(program.is_ok());

        let loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(
            loader.as_ref(),
            Some(ProgramVerification::InternalConsistency),
        );
        assert!(program.is_ok());

        let loader = BufferDataLoader::new(&[]);
        let program = Program::load(loader.as_ref(), None);
        assert!(program.is_err());
    }

    #[test]
    fn num_methods() {
        let loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(loader.as_ref(), None).unwrap();
        assert_eq!(program.num_methods(), 1);
    }

    #[test]
    fn get_method_name() {
        let loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(loader.as_ref(), None).unwrap();
        assert_eq!(program.get_method_name(0).ok(), Some("forward"));
        assert_eq!(program.get_method_name(1).ok(), None);
    }

    #[test]
    fn method_meta() {
        let loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(loader.as_ref(), None).unwrap();
        let method_meta = program.method_meta(c"forward").unwrap();
        assert!(program.method_meta(c"non-existing-method").is_err());

        assert_eq!(method_meta.name(), "forward");

        assert_eq!(method_meta.num_inputs(), 2);
        assert_eq!(method_meta.input_tag(0).unwrap(), Tag::Tensor);
        assert_eq!(method_meta.input_tag(1).unwrap(), Tag::Tensor);
        assert!(method_meta.input_tag(2).is_err());
        let tinfo1 = method_meta.input_tensor_meta(1).unwrap();
        let tinfo2 = method_meta.input_tensor_meta(0).unwrap();
        for tinfo in [tinfo1, tinfo2] {
            assert_eq!(tinfo.sizes(), &[1]);
            assert_eq!(tinfo.dim_order(), &[0]);
            assert_eq!(tinfo.scalar_type(), ScalarType::Float);
            assert!(tinfo.is_memory_planned());
            assert_eq!(tinfo.nbytes(), 4);
        }

        assert_eq!(method_meta.num_outputs(), 1);
        assert_eq!(method_meta.output_tag(0).unwrap(), Tag::Tensor);
        assert!(method_meta.output_tag(1).is_err());
        let tinfo = method_meta.output_tensor_meta(0).unwrap();
        assert_eq!(tinfo.sizes(), &[1]);
        assert_eq!(tinfo.dim_order(), &[0]);
        assert_eq!(tinfo.scalar_type(), ScalarType::Float);
        assert!(tinfo.is_memory_planned());
        assert_eq!(tinfo.nbytes(), 4);
        assert!(method_meta.output_tensor_meta(1).is_err());

        assert_eq!(method_meta.num_attributes(), 0);
        assert!(method_meta.attribute_tensor_meta(0).is_err());

        for i in 0..method_meta.num_memory_planned_buffers() {
            assert!(method_meta.memory_planned_buffer_size(i).is_ok());
        }
        assert!(method_meta
            .memory_planned_buffer_size(method_meta.num_memory_planned_buffers())
            .is_err());

        for i in 0..method_meta.num_backends() {
            let backend_name = method_meta.get_backend_name(i).unwrap();
            assert!(!backend_name.is_empty());
            #[cfg(feature = "alloc")]
            assert!(
                method_meta.uses_backend(std::ffi::CString::new(backend_name).unwrap().as_c_str())
            );
        }
        assert!(!method_meta.uses_backend(c"non-existing-backend"));
    }

    #[cfg(tests_with_kernels)]
    #[test]
    fn load_method() {
        use crate::memory::{BufferMemoryAllocator, HierarchicalAllocator};
        use crate::util::Span;

        let mut buffer = [0_u8; 4096];
        let allocator = BufferMemoryAllocator::new(&mut buffer);

        let data_loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(
            data_loader.as_ref(),
            Some(ProgramVerification::InternalConsistency),
        )
        .unwrap();

        let method_meta = program.method_meta(c"forward").unwrap();
        let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
        let planned_arenas = allocator
            .allocate_arr_fn(num_memory_planned_buffers, |idx| {
                let buf_size = method_meta.memory_planned_buffer_size(idx).unwrap();
                Span::from_slice(allocator.allocate_arr::<u8>(buf_size).unwrap())
            })
            .unwrap();

        let mut planned_memory = HierarchicalAllocator::new(planned_arenas);
        let memory_manager = MemoryManager::new(&allocator, Some(&mut planned_memory), None);

        assert!(program
            .load_method(c"non-existing-method", &memory_manager, None, None)
            .is_err());
        assert!(program
            .load_method(c"forward", &memory_manager, None, None)
            .is_ok());
    }

    #[test]
    fn check_header() {
        assert_ne!(Program::check_header(&[]), HeaderStatus::CompatibleVersion);
        assert_ne!(
            Program::check_header(&[42, 6, 17]),
            HeaderStatus::CompatibleVersion
        );
        assert_eq!(
            Program::check_header(ADD_MODEL_BYTES),
            HeaderStatus::CompatibleVersion
        );
    }

    #[cfg(tests_with_kernels)]
    #[test]
    fn method_execution() {
        use crate::memory::{BufferMemoryAllocator, HierarchicalAllocator};
        use crate::tensor::{Tensor, TensorImpl};
        use crate::util::Span;

        let mut buffer = [0_u8; 4096];
        let allocator = BufferMemoryAllocator::new(&mut buffer);

        let data_loader = BufferDataLoader::new(ADD_MODEL_BYTES);
        let program = Program::load(
            data_loader.as_ref(),
            Some(ProgramVerification::InternalConsistency),
        )
        .unwrap();

        let method_meta = program.method_meta(c"forward").unwrap();
        let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
        let planned_arenas = allocator
            .allocate_arr_fn(num_memory_planned_buffers, |idx| {
                let buf_size = method_meta.memory_planned_buffer_size(idx).unwrap();
                Span::from_slice(allocator.allocate_arr::<u8>(buf_size).unwrap())
            })
            .unwrap();

        let mut planned_memory = HierarchicalAllocator::new(planned_arenas);
        let memory_manager = MemoryManager::new(&allocator, Some(&mut planned_memory), None);

        let mut method = program
            .load_method(c"forward", &memory_manager, None, None)
            .unwrap();
        assert_eq!(method.inputs_size(), 2);
        assert!(method.get_attribute("non-existing-attr").is_err());
        let execution = method.start_execution();
        assert!(matches!(
            execution.execute(), // inputs not set
            Err(Error::InvalidArgument)
        ));
        let mut execution = method.start_execution();

        let sizes = [1];
        let data = [1.0_f32];
        let dim_order = [0];
        let strides = [1];
        let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();
        let tensor = Tensor::new_in_allocator(&tensor_impl, &allocator);
        let input1 = EValue::new_in_allocator(tensor, &allocator);

        let sizes = [1];
        let data = [1.0_f32];
        let dim_order = [0];
        let strides = [1];
        let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();
        let tensor = Tensor::new_in_allocator(&tensor_impl, &allocator);
        let input2 = EValue::new_in_allocator(tensor, &allocator);

        assert!(execution.set_input(&input1, 2).is_err());
        execution.set_input(&input1, 0).unwrap();
        execution.set_input(&input2, 1).unwrap();
        let outputs = execution.execute().unwrap();

        assert!(!outputs.is_empty());
        assert_eq!(outputs.len(), 1);
        let output = outputs.get(0);
        assert_eq!(output.tag(), Tag::Tensor);
        let output = output.as_tensor().into_typed::<f32>();
        assert_eq!(output.sizes(), [1]);
        assert_eq!(output[&[0]], 2.0);
    }
}
