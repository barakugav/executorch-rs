//! A higher-level for simple execution of programs.
//!
//! This module provides a higher-level interface for loading programs and executing methods within them.
//! Compared to the lower-level [`program`](crate::program) interface, the [`module`](crate::module) interface is more
//! user-friendly, uses the default memory allocator, and provides automatic memory management.
//!
//! This module is enabled by the `module` feature.
//!
//! See the `hello_world_add` example for how to load and execute a module.

use std::collections::HashSet;
use std::path::Path;
use std::ptr;

use crate::error::{fallible, Result};
use crate::et_rs_c::VecChar;
use crate::evalue::EValue;
use crate::program::{MethodMeta, ProgramVerification};
use crate::util::cpp_vec::CppVecImpl;
use crate::util::{self, ArrayRef, Destroy, IntoRust, NonTriviallyMovable, NonTriviallyMovableVec};
use crate::{et_c, et_rs_c};

/// A facade class for loading programs and executing methods within them.
///
/// See the `hello_world_add` example for how to load and execute a module.
pub struct Module(NonTriviallyMovable<'static, et_c::extension::Module>);
impl Module {
    /// Constructs an instance by loading a program from a file with specified
    /// memory locking behavior.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the ExecuTorch program file to load.
    /// * `mlock_config` - The memory locking configuration to use. Defaults to `MlockConfig::UseMlock`.
    ///
    /// # Returns
    ///
    /// A new instance of Module.
    ///
    /// # Panics
    ///
    /// If the file path is not a valid UTF-8 string or contains a null character.
    pub fn new(file_path: impl AsRef<Path>, mlock_config: Option<LoadMode>) -> Self {
        let file_path = file_path.as_ref().to_str().unwrap();
        let file_path = ArrayRef::from_slice(util::str2chars(file_path).unwrap());
        let mlock_config = mlock_config.unwrap_or(LoadMode::MmapUseMlock);
        let event_tracer = ptr::null_mut(); // TODO: support event tracer
        Self(unsafe {
            NonTriviallyMovable::new_boxed(|p| {
                et_rs_c::Module_new(p, file_path.0, mlock_config, event_tracer)
            })
        })
    }

    /// Loads the program using the specified data loader and memory allocator.
    ///
    /// # Arguments
    ///
    /// * `verification` - The type of verification to do before returning success.
    /// Defaults to `ProgramVerification::Minimal`.
    ///
    /// # Returns
    ///
    /// An Error to indicate success or failure of the loading process.
    pub fn load(&mut self, verification: Option<ProgramVerification>) -> Result<()> {
        let verification = verification.unwrap_or(ProgramVerification::Minimal);
        unsafe { et_c::extension::Module_load(self.0.as_mut().unwrap(), verification) }.rs()
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
        let names = fallible(|names| unsafe {
            et_rs_c::Module_method_names(self.0.as_mut().unwrap(), names)
        })?
        .rs();
        Ok(names
            .as_slice()
            .iter()
            .map(|s: &VecChar| util::chars2string(s.as_slice().to_vec()))
            .collect())
    }

    /// Load a specific method from the program and set up memory management if needed.
    /// The loaded method is cached to reuse the next time it's executed.
    ///
    /// # Arguments
    ///
    /// * `method_name` - The name of the method to load.
    ///
    /// # Returns
    ///
    /// An Error to indicate success or failure.
    ///
    /// # Panics
    ///
    /// If the method name is not a valid UTF-8 string or contains a null character.
    pub fn load_method(&mut self, method_name: impl AsRef<str>) -> Result<()> {
        let method_name = ArrayRef::from_slice(util::str2chars(method_name.as_ref()).unwrap());
        unsafe { et_rs_c::Module_load_method(self.0.as_mut().unwrap(), method_name.0) }.rs()
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
        let method_name = ArrayRef::from_slice(util::str2chars(method_name.as_ref()).unwrap());
        unsafe { et_rs_c::Module_is_method_loaded(self.0.as_ref(), method_name.0) }
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
    pub fn method_meta(&self, method_name: impl AsRef<str>) -> Result<MethodMeta> {
        let method_name = ArrayRef::from_slice(util::str2chars(method_name.as_ref()).unwrap());
        let meta = fallible(|meta| unsafe {
            et_rs_c::Module_method_meta(self.0.as_ref() as *const _ as *mut _, method_name.0, meta)
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
    pub fn execute<'a>(
        &'a mut self,
        method_name: impl AsRef<str>,
        inputs: &[EValue],
    ) -> Result<Vec<EValue<'a>>> {
        let method_name = ArrayRef::from_slice(util::str2chars(method_name.as_ref()).unwrap());
        let inputs = unsafe {
            NonTriviallyMovableVec::new(inputs.len(), |i, p| {
                et_rs_c::EValue_copy(inputs[i].as_evalue(), p.as_mut_ptr())
            })
        };
        let inputs = ArrayRef::from_slice(inputs.as_slice());
        let mut outputs = fallible(|outputs| unsafe {
            et_rs_c::Module_execute(self.0.as_mut().unwrap(), method_name.0, inputs.0, outputs)
        })?
        .rs();
        Ok(outputs
            .as_mut_slice()
            .iter_mut()
            .map(|val| unsafe { EValue::move_from(val) })
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
    pub fn forward<'a>(&'a mut self, inputs: &[EValue]) -> Result<Vec<EValue<'a>>> {
        self.execute("forward", inputs)
    }
}
impl Destroy for et_c::extension::Module {
    unsafe fn destroy(&mut self) {
        unsafe { et_rs_c::Module_destructor(self) }
    }
}

/// Enum to define loading behavior.
pub type LoadMode = et_c::extension::Module_LoadMode;
