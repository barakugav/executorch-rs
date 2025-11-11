//! A higher-level for simple execution of programs.
//!
//! This module provides a higher-level interface for loading programs and executing methods within them.
//! Compared to the lower-level [`program`](crate::program) interface, the [`module`](crate::module) interface is more
//! user-friendly, uses the default memory allocator, and provides automatic memory management.
//!
//! This module is enabled by the `module` feature.
//!
//! See the `hello_world` example for how to load and execute a module.

use core::marker::PhantomData;
use std::collections::HashSet;
use std::path::Path;

use executorch_sys as sys;

use crate::evalue::EValue;
use crate::event_tracer::{EventTracer, EventTracerPtr};
use crate::memory::HierarchicalAllocator;
use crate::program::{MethodMeta, ProgramVerification};
use crate::util::{try_c_new, ArrayRef, IntoCpp, IntoRust, NonTriviallyMovableVec};
use crate::{Error, Result};

/// A facade class for loading programs and executing methods within them.
///
/// See the `hello_world` example for how to load and execute a module.
pub struct Module<'a>(sys::cxx::UniquePtr<sys::cpp::Module>, PhantomData<&'a ()>);
impl<'a> Module<'a> {
    /// Constructs an instance by loading a program from a file with specified
    /// memory locking behavior.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the ExecuTorch program file to load.
    /// - `data_files` - The path to one or more .ptd file/s.
    /// * `load_mode` - The loading mode to use. Defaults to `LoadMode::File`.
    /// * `event_tracer` - A EventTracer used for tracking and logging events.
    ///
    /// # Returns
    ///
    /// A new instance of Module.
    ///
    /// # Panics
    ///
    /// May panic if any of the file path or the data map path are not a valid UTF-8 string or contains a null character.
    pub fn new(
        file_path: &Path,
        data_files: &[&Path],
        load_mode: Option<LoadMode>,
        event_tracer: Option<EventTracerPtr<'a>>,
    ) -> Self {
        let data_files = data_files
            .iter()
            .map(|f| f.as_os_str().to_str().ok_or(Error::InvalidString).unwrap())
            .collect::<Vec<_>>();
        sys::cxx::let_cxx_string!(file_path = file_path.as_os_str().as_encoded_bytes());

        let load_mode = load_mode.unwrap_or(LoadMode::File).cpp();
        let event_tracer = event_tracer
            .map(|tracer| tracer.0)
            .unwrap_or(sys::cxx::UniquePtr::null());
        let module = sys::cpp::Module_new(&file_path, &data_files, load_mode, event_tracer);
        Self(module, PhantomData)
    }

    /// Constructs an instance by loading a program from a file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the ExecuTorch program file to load.
    ///
    /// # Returns
    ///
    /// A new instance of Module.
    ///
    /// # Panics
    ///
    /// If the file path is not a valid UTF-8 string or contains a null character.
    pub fn from_file_path(file_path: impl AsRef<Path>) -> Self {
        Self::new(file_path.as_ref(), &[], None, None)
    }

    /// Loads the program using the specified data loader and memory allocator.
    ///
    /// # Arguments
    ///
    /// * `verification` - The type of verification to do before returning success.
    ///   Defaults to `ProgramVerification::Minimal`.
    ///
    /// # Returns
    ///
    /// An Error to indicate success or failure of the loading process.
    pub fn load(&mut self, verification: Option<ProgramVerification>) -> Result<()> {
        let verification = verification.unwrap_or(ProgramVerification::Minimal).cpp();
        sys::cpp::Module_load(self.0.as_mut().unwrap(), verification).rs()
    }

    /// Checks if the program is loaded.
    pub fn is_loaded(&self) -> bool {
        sys::cpp::Module_is_loaded(self.0.as_ref().unwrap())
    }

    /// Get the number of methods available in the loaded program.
    pub fn num_methods(&mut self) -> Result<usize> {
        // Safety: sys::Module_num_methods writes to the pointer.
        unsafe {
            try_c_new(|num_methods| {
                sys::cpp::Module_num_methods(self.0.as_mut().unwrap(), num_methods)
            })
        }
    }

    /// Get a list of method names available in the loaded program.
    /// Loads the program and method if needed.
    ///
    /// # Returns
    ///
    /// A set of strings containing the names of the methods, or an error if the program or method failed to load.
    pub fn method_names(&mut self) -> Result<HashSet<String>> {
        let self_ = self.0.as_mut().unwrap();
        let names = unsafe { try_c_new(|names| sys::cpp::Module_method_names(self_, names)) }?;
        Ok(names.into_iter().map(|s| s.to_string()).collect())
    }

    /// Load a specific method from the program and set up memory management if needed.
    /// The loaded method is cached to reuse the next time it's executed.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to load.
    /// * `planned_memory` - The memory-planned buffers to use for mutable tensor data when executing a method.
    /// * `event_tracer` - Per-method event tracer to profile/trace methods individually. When not given, the event
    ///   tracer passed to the Module constructor is used. Otherwise, this per-method event tracer takes precedence.
    ///
    /// # Returns
    ///
    /// An Error to indicate success or failure.
    ///
    /// # Panics
    ///
    /// May panic if the method name is not a valid UTF-8 string or contains a null character.
    pub fn load_method(
        &mut self,
        method_name: &str,
        planned_memory: Option<&'a mut HierarchicalAllocator>,
        event_tracer: Option<&'a mut EventTracer>,
    ) -> Result<()> {
        sys::cxx::let_cxx_string!(method_name = method_name);
        let event_tracer = event_tracer
            .map(|tracer| tracer as *mut EventTracer as *mut sys::cpp::EventTracer)
            .unwrap_or(std::ptr::null_mut());
        let planned_memory = planned_memory
            .map(|allocator| (&mut allocator.0) as *mut sys::cpp::HierarchicalAllocator)
            .unwrap_or(std::ptr::null_mut());
        unsafe {
            sys::cpp::Module_load_method(
                self.0.as_mut().unwrap(),
                &method_name,
                planned_memory,
                event_tracer,
            )
            .rs()
        }
    }

    /// Unload a specific method from the program.
    ///
    /// # Arguments
    ///
    /// - `method_name`: The name of the method to unload.
    ///
    /// # Returns
    ///
    /// True if the method is unloaded, false if no-op.
    ///
    /// # Panics
    ///
    /// May panic if the method name is not a valid UTF-8 string or contains a null character.
    pub fn unload_method(&mut self, method_name: &str) -> bool {
        sys::cxx::let_cxx_string!(method_name = method_name);
        unsafe { sys::cpp::Module_unload_method(self.0.as_mut().unwrap(), &method_name) }
    }

    /// Checks if a specific method is loaded.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to check.
    ///
    /// # Returns
    ///
    /// true if the method specified by method_name is loaded, false otherwise.
    ///
    /// # Panics
    ///
    /// May panic if the method name is not a valid UTF-8 string or contains a null character.
    pub fn is_method_loaded(&self, method_name: &str) -> bool {
        sys::cxx::let_cxx_string!(method_name = method_name);
        sys::cpp::Module_is_method_loaded(self.0.as_ref().unwrap(), &method_name)
    }

    /// Get a method metadata struct by method name.
    ///
    /// Loads the program if needed.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to get the metadata for.
    ///
    /// # Returns
    ///
    /// A method metadata, or an error if the program or method failed to load.
    ///
    /// # Panics
    ///
    /// May panic if the method name contains a null character.
    pub fn method_meta<'b>(&'b mut self, method_name: &str) -> Result<MethodMeta<'b>> {
        sys::cxx::let_cxx_string!(method_name = method_name);
        // Safety: sys::Module_method_meta writes to the pointer.
        let meta = unsafe {
            try_c_new(|meta| {
                sys::cpp::Module_method_meta(self.0.as_mut().unwrap(), &method_name, meta)
            })?
        };
        // Safety: the method metadata is valid as long as self is valid
        Ok(unsafe { MethodMeta::new(meta) })
    }

    /// Executes a specific method with the given input and retrieves the output.
    /// Loads the program and method before executing if needed.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to execute.
    /// * `inputs` - A slice of input values to be passed to the method.
    ///
    /// # Returns
    ///
    /// A result object containing either a vector of output values from the method or an error to indicate failure.
    ///
    /// # Panics
    ///
    /// May panic if the method name is not a valid UTF-8 string or contains a null character.
    pub fn execute<'b>(
        &'b mut self,
        method_name: &str,
        inputs: &[EValue],
    ) -> Result<Vec<EValue<'b>>> {
        sys::cxx::let_cxx_string!(method_name = method_name);
        let inputs = unsafe {
            NonTriviallyMovableVec::<sys::EValueStorage>::new(inputs.len(), |i, p| {
                sys::executorch_EValue_copy(
                    inputs[i].cpp(),
                    sys::EValueRefMut {
                        ptr: p.as_mut_ptr() as *mut _,
                    },
                )
            })
        };
        let inputs = ArrayRef::from_slice(inputs.as_slice());
        // Safety: sys::Module_execute writes to the pointer.
        let mut outputs = unsafe {
            try_c_new(|outputs| {
                sys::cpp::Module_execute(self.0.as_mut().unwrap(), &method_name, inputs.0, outputs)
            })?
            .rs()
        };
        Ok(outputs
            .as_mut_slice()
            .iter_mut()
            .map(|val| unsafe {
                EValue::move_from(sys::EValueRefMut {
                    ptr: val as *mut sys::EValueStorage as *mut _,
                })
            })
            .collect())
    }

    /// Executes the 'forward' method with the given input and retrieves the output.
    /// Loads the program and method before executing if needed.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input values for the 'forward' method.
    ///
    /// # Returns
    ///
    /// A result object containing either a vector of output values from the 'forward' method or an error to indicate failure.
    pub fn forward<'b>(&'b mut self, inputs: &[EValue]) -> Result<Vec<EValue<'b>>> {
        self.execute("forward", inputs)
    }
}
unsafe impl Send for Module<'_> {}

#[repr(u32)]
#[doc = " Enum to define loading behavior."]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum LoadMode {
    #[doc = " Load the whole file as a buffer."]
    File = sys::ModuleLoadMode::ModuleLoadMode_File as u32,
    #[doc = " Use mmap to load pages into memory."]
    Mmap = sys::ModuleLoadMode::ModuleLoadMode_Mmap as u32,
    #[doc = " Use memory locking and handle errors."]
    MmapUseMlock = sys::ModuleLoadMode::ModuleLoadMode_MmapUseMlock as u32,
    #[doc = " Use memory locking and ignore errors."]
    MmapUseMlockIgnoreErrors = sys::ModuleLoadMode::ModuleLoadMode_MmapUseMlockIgnoreErrors as u32,
}
impl IntoCpp for LoadMode {
    type CppType = sys::ModuleLoadMode;
    fn cpp(self) -> Self::CppType {
        match self {
            LoadMode::File => sys::ModuleLoadMode::ModuleLoadMode_File,
            LoadMode::Mmap => sys::ModuleLoadMode::ModuleLoadMode_Mmap,
            LoadMode::MmapUseMlock => sys::ModuleLoadMode::ModuleLoadMode_MmapUseMlock,
            LoadMode::MmapUseMlockIgnoreErrors => {
                sys::ModuleLoadMode::ModuleLoadMode_MmapUseMlockIgnoreErrors
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::add_model_path;

    #[test]
    fn load() {
        for load_mode in [
            None,
            Some(LoadMode::File),
            Some(LoadMode::Mmap),
            Some(LoadMode::MmapUseMlock),
            Some(LoadMode::MmapUseMlockIgnoreErrors),
        ] {
            for verification in [
                None,
                Some(ProgramVerification::Minimal),
                Some(ProgramVerification::InternalConsistency),
            ] {
                // TODO: test with data files (.ptd)
                let mut module = Module::new(&add_model_path(), &[], load_mode, None);
                assert!(!module.is_loaded());
                assert!(module.load(verification).is_ok());
                assert!(module.is_loaded());
            }
        }

        let mut module = Module::from_file_path("non-existing-file.pte2");
        assert!(!module.is_loaded());
        assert!(module.load(None).is_err());
        assert!(!module.is_loaded());
    }

    #[test]
    fn method_names() {
        let mut module = Module::from_file_path(add_model_path());
        let names = module.method_names().unwrap();
        assert_eq!(names, HashSet::from_iter(["forward".to_string()]));

        let mut module = Module::from_file_path("non-existing-file.pte2");
        assert!(module.method_names().is_err());
    }

    #[test]
    fn num_methods() {
        let mut module = Module::from_file_path(add_model_path());
        let num_methods = module.num_methods().unwrap();
        assert_eq!(num_methods, 1);

        let mut module = Module::from_file_path("non-existing-file.pte2");
        assert!(module.num_methods().is_err());
    }

    #[cfg(tests_with_kernels)]
    #[test]
    fn load_method() {
        let mut module = Module::from_file_path(add_model_path());
        assert!(!module.is_method_loaded("forward"));
        assert!(module.load_method("forward", None, None).is_ok());
        assert!(module.is_method_loaded("forward"));
        assert!(module
            .load_method("non-existing-method", None, None)
            .is_err());
        assert!(!module.is_method_loaded("non-existing-method"));

        let mut module = Module::from_file_path("non-existing-file.pte2");
        assert!(module.load_method("forward", None, None).is_err());
    }

    #[cfg(tests_with_kernels)]
    #[test]
    fn unload_method() {
        let mut module = Module::from_file_path(add_model_path());
        assert!(!module.is_method_loaded("forward"));
        assert!(module.load_method("forward", None, None).is_ok());
        assert!(module.is_method_loaded("forward"));
        assert!(!module.unload_method("non-existing-method"));
        assert!(module.is_method_loaded("forward"));
        assert!(module.unload_method("forward"));
        assert!(!module.is_method_loaded("forward"));
    }

    #[cfg(tests_with_kernels)]
    #[test]
    fn method_meta() {
        use crate::evalue::Tag;
        use crate::tensor::ScalarType;

        let mut module = Module::from_file_path(add_model_path());
        let method_meta = module.method_meta("forward").unwrap();

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
            assert_eq!(tinfo.nbytes(), 4);
        }

        assert_eq!(method_meta.num_outputs(), 1);
        assert_eq!(method_meta.output_tag(0).unwrap(), Tag::Tensor);
        assert!(method_meta.output_tag(1).is_err());
        let tinfo = method_meta.output_tensor_meta(0).unwrap();
        assert_eq!(tinfo.sizes(), &[1]);
        assert_eq!(tinfo.dim_order(), &[0]);
        assert_eq!(tinfo.scalar_type(), ScalarType::Float);
        assert_eq!(tinfo.nbytes(), 4);
        assert!(method_meta.output_tensor_meta(1).is_err());

        for i in 0..method_meta.num_memory_planned_buffers() {
            assert!(method_meta.memory_planned_buffer_size(i).is_ok());
        }
        assert!(method_meta
            .memory_planned_buffer_size(method_meta.num_memory_planned_buffers())
            .is_err());

        let mut module = Module::from_file_path("non-existing-file.pte2");
        assert!(module.method_meta("forward").is_err());
    }

    #[cfg(tests_with_kernels)]
    #[test]
    fn execute() {
        use crate::evalue::Tag;
        use crate::tensor::{Tensor, TensorImpl};

        let mut module = Module::from_file_path(add_model_path());

        let sizes = [1];
        let data = [1.0_f32];
        let dim_order = [0];
        let strides = [1];
        let tensor1 = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();

        let sizes = [1];
        let data = [1.0_f32];
        let dim_order = [0];
        let strides = [1];
        let tensor2 = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();

        for i in 0..2 {
            let inputs = [
                EValue::new(Tensor::new(&tensor1)),
                EValue::new(Tensor::new(&tensor2)),
            ];
            let outputs = if i == 0 {
                module.execute("forward", &inputs).unwrap()
            } else {
                module.forward(&inputs).unwrap()
            };
            assert_eq!(outputs.len(), 1);
            let output = &outputs[0];
            assert_eq!(output.tag(), Tag::Tensor);
            let output = output.as_tensor().into_typed::<f32>();
            assert_eq!(output.sizes(), [1]);
            assert_eq!(output[&[0]], 2.0);
        }

        // wrong number of inputs
        let inputs = [EValue::new(Tensor::new(&tensor1))];
        assert!(module.execute("forward", &inputs).is_err());
        let inputs = [
            EValue::new(Tensor::new(&tensor1)),
            EValue::new(Tensor::new(&tensor2)),
            EValue::new(Tensor::new(&tensor2)),
        ];
        assert!(module.execute("forward", &inputs).is_err());

        // non-existing method
        let inputs = [
            EValue::new(Tensor::new(&tensor1)),
            EValue::new(Tensor::new(&tensor2)),
        ];
        assert!(module.execute("non-existing-method", &inputs).is_err());

        // non-existing file
        let mut module = Module::from_file_path("non-existing-file.pte2");
        let inputs = [
            EValue::new(Tensor::new(&tensor1)),
            EValue::new(Tensor::new(&tensor2)),
        ];
        assert!(module.execute("non-existing-method", &inputs).is_err());
    }
}
