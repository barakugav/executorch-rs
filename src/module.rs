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

use crate::evalue::EValue;
use crate::event_tracer::{EventTracer, EventTracerPtr};
use crate::program::{MethodMeta, ProgramVerification};
use crate::util::{try_c_new, ArrayRef, IntoCpp, IntoRust, NonTriviallyMovableVec};
use crate::{Error, Result};
use executorch_sys as et_c;

/// A facade class for loading programs and executing methods within them.
///
/// See the `hello_world` example for how to load and execute a module.
pub struct Module<'a>(
    executorch_sys::cxx::UniquePtr<et_c::cpp::Module>,
    PhantomData<&'a ()>,
);
impl<'a> Module<'a> {
    /// Constructs an instance by loading a program from a file with specified
    /// memory locking behavior.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the ExecuTorch program file to load.
    /// * `load_mode` - The loading mode to use. Defaults to `LoadMode::MmapUseMlock`.
    /// * `event_tracer` - A EventTracer used for tracking and logging events.
    ///
    /// # Returns
    ///
    /// A new instance of Module.
    ///
    /// # Panics
    ///
    /// If the file path is not a valid UTF-8 string or contains a null character.
    pub fn new(
        file_path: impl AsRef<Path>,
        load_mode: Option<LoadMode>,
        event_tracer: Option<EventTracerPtr<'a>>,
    ) -> Self {
        let file_path = file_path.as_ref().to_str().ok_or(Error::ToCStr).unwrap();
        let load_mode = load_mode.unwrap_or(LoadMode::MmapUseMlock).cpp();
        let event_tracer = event_tracer
            .map(|tracer| tracer.0)
            .unwrap_or(et_c::cxx::UniquePtr::null());
        let module = et_c::cpp::Module_new(file_path, load_mode, event_tracer);
        Self(module, PhantomData)
    }

    /// Loads the program using the specified data loader and memory allocator.
    ///
    /// # Arguments
    ///
    /// * `verification` - The type of verification to do before returning success.
    ///     Defaults to `ProgramVerification::Minimal`.
    ///
    /// # Returns
    ///
    /// An Error to indicate success or failure of the loading process.
    pub fn load(&mut self, verification: Option<ProgramVerification>) -> Result<()> {
        let verification = verification.unwrap_or(ProgramVerification::Minimal).cpp();
        et_c::cpp::Module_load(self.0.as_mut().unwrap(), verification).rs()
    }

    // /// Checks if the program is loaded.
    // ///
    // /// # Returns
    // ///
    // /// true if the program is loaded, false otherwise.
    // pub fn is_loaded(&self) -> bool {
    //     unsafe { et_c::extension::Module_is_loaded(self.0.as_ref()) }
    // }

    /// Get a list of method names available in the loaded program.
    /// Loads the program and method if needed.
    ///
    /// # Returns
    ///
    /// A set of strings containing the names of the methods, or an error if the program or method failed to load.
    pub fn method_names(&mut self) -> Result<HashSet<String>> {
        let mut names = Vec::new();
        let self_ = self.0.as_mut().unwrap();
        unsafe { et_c::cpp::Module_method_names(self_, &mut names).rs()? };
        Ok(names.into_iter().map(|s| s.to_string()).collect())
    }

    /// Load a specific method from the program and set up memory management if needed.
    /// The loaded method is cached to reuse the next time it's executed.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to load.
    /// * `event_tracer` - The event tracer to use for this method run.
    ///
    /// # Returns
    ///
    /// An Error to indicate success or failure.
    ///
    /// # Panics
    ///
    /// If the method name is not a valid UTF-8 string or contains a null character.
    pub fn load_method(
        &mut self,
        method_name: impl AsRef<str>,
        event_tracer: Option<&'a mut EventTracer>,
    ) -> Result<()> {
        let event_tracer = event_tracer
            .map(|tracer| tracer as *mut EventTracer as *mut et_c::cpp::EventTracer)
            .unwrap_or(std::ptr::null_mut());
        unsafe {
            et_c::cpp::Module_load_method(
                self.0.as_mut().unwrap(),
                method_name.as_ref(),
                event_tracer,
            )
            .rs()
        }
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
    /// If the method name is not a valid UTF-8 string or contains a null character.
    pub fn is_method_loaded(&self, method_name: impl AsRef<str>) -> bool {
        et_c::cpp::Module_is_method_loaded(self.0.as_ref().unwrap(), method_name.as_ref())
    }

    /// Get a method metadata struct by method name.
    /// Loads the program and method if needed.
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
    /// If the method name is not a valid UTF-8 string or contains a null character.
    pub fn method_meta(&mut self, method_name: impl AsRef<str>) -> Result<MethodMeta> {
        let meta = try_c_new(|meta| unsafe {
            et_c::cpp::Module_method_meta(self.0.as_mut().unwrap(), method_name.as_ref(), meta)
        })?;
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
    /// If the method name is not a valid UTF-8 string or contains a null character.
    pub fn execute<'b>(
        &'b mut self,
        method_name: impl AsRef<str>,
        inputs: &[EValue],
    ) -> Result<Vec<EValue<'b>>> {
        let inputs = unsafe {
            NonTriviallyMovableVec::<et_c::EValueStorage>::new(inputs.len(), |i, p| {
                et_c::executorch_EValue_copy(
                    inputs[i].cpp(),
                    et_c::EValueRefMut {
                        ptr: p.as_mut_ptr() as *mut _,
                    },
                )
            })
        };
        let inputs = ArrayRef::from_slice(inputs.as_slice());
        let mut outputs = try_c_new(|outputs| unsafe {
            et_c::cpp::Module_execute(
                self.0.as_mut().unwrap(),
                method_name.as_ref(),
                inputs.0,
                outputs,
            )
        })?
        .rs();
        Ok(outputs
            .as_mut_slice()
            .iter_mut()
            .map(|val| unsafe {
                EValue::move_from(et_c::EValueRefMut {
                    ptr: val as *mut et_c::EValueStorage as *mut _,
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

#[repr(u32)]
#[doc = " Enum to define loading behavior."]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum LoadMode {
    #[doc = " Load the whole file as a buffer."]
    File = 0,
    #[doc = " Use mmap to load pages into memory."]
    Mmap = 1,
    #[doc = " Use memory locking and handle errors."]
    MmapUseMlock = 2,
    #[doc = " Use memory locking and ignore errors."]
    MmapUseMlockIgnoreErrors = 3,
}
impl IntoCpp for LoadMode {
    type CppType = et_c::ModuleLoadMode;
    fn cpp(self) -> Self::CppType {
        match self {
            LoadMode::File => et_c::ModuleLoadMode::ModuleLoadMode_File,
            LoadMode::Mmap => et_c::ModuleLoadMode::ModuleLoadMode_Mmap,
            LoadMode::MmapUseMlock => et_c::ModuleLoadMode::ModuleLoadMode_MmapUseMlock,
            LoadMode::MmapUseMlockIgnoreErrors => {
                et_c::ModuleLoadMode::ModuleLoadMode_MmapUseMlockIgnoreErrors
            }
        }
    }
}
