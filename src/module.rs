use std::collections::HashSet;
use std::path::Path;
use std::ptr;

use crate::{et_c, et_rs_c, util::IntoRust, EValue, Result, Span};
use crate::{util, MethodMeta, ProgramVerification};

/// A facade class for loading programs and executing methods within them.
pub struct Module(et_c::Module);
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
    pub fn new(file_path: impl AsRef<Path>, mlock_config: Option<MlockConfig>) -> Self {
        let file_path = file_path.as_ref().to_str().unwrap();
        let file_path = Span::new(util::str2chars(file_path).unwrap());
        let mlock_config = mlock_config.unwrap_or(MlockConfig::UseMlock);
        let event_tracer = ptr::null_mut(); // TODO: support event tracer
        Self(unsafe { et_rs_c::Module_new(file_path.0, mlock_config, event_tracer) })
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
        unsafe { et_c::Module_load(&mut self.0, verification) }.rs()
    }

    /// Checks if the program is loaded.
    ///
    /// # Returns
    ///
    /// true if the program is loaded, false otherwise.
    pub fn is_loaded(&self) -> bool {
        unsafe { et_c::Module_is_loaded(&self.0) }
    }

    /// Get a list of method names available in the loaded program.
    /// Loads the program and method if needed.
    ///
    /// # Returns
    ///
    /// A set of strings containing the names of the methods, or an error if the program or method failed to load.
    pub fn method_names(&mut self) -> Result<HashSet<String>> {
        let names = unsafe { et_rs_c::Module_method_names(&mut self.0) }.rs()?;
        Ok(names
            .rs()
            .into_iter()
            .map(|s| util::chars2string(s.rs()))
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
    pub fn load_method(&mut self, method_name: &str) -> Result<()> {
        let method_name = Span::new(util::str2chars(method_name).unwrap());
        unsafe { et_rs_c::Module_load_method(&mut self.0, method_name.0) }.rs()
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
    pub fn is_method_loaded(&self, method_name: &str) -> bool {
        let method_name = Span::new(util::str2chars(method_name).unwrap());
        unsafe { et_rs_c::Module_is_method_loaded(&self.0, method_name.0) }
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
    pub fn method_meta<'a>(&'a self, method_name: &str) -> Result<MethodMeta<'a>> {
        let method_name = Span::new(util::str2chars(method_name).unwrap());
        let meta = unsafe { et_rs_c::Module_method_meta(&self.0, method_name.0) }.rs()?;
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
        method_name: &str,
        inputs: &[EValue],
    ) -> Result<Vec<EValue<'a>>> {
        let method_name = Span::new(util::str2chars(method_name).unwrap());
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
impl Drop for Module {
    fn drop(&mut self) {
        unsafe { et_rs_c::Module_destructor(&mut self.0) };
    }
}

pub type MlockConfig = et_c::Module_MlockConfig;
