// Clippy doesnt detect the 'Safety' comments in the cxx bridge.
#![allow(clippy::missing_safety_doc)]

use cxx::{type_id, ExternType};

#[cxx::bridge]
pub(crate) mod ffi {

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        /// Cpp executorch error type.
        type Error = crate::Error;

        /// Types of validation that a `Program` can do before parsing the data.
        type ProgramVerification = crate::ProgramVerification;

        /// Describes a method in an ExecuTorch program.
        ///
        /// The program used to create a MethodMeta object must outlive the MethodMeta.
        /// It is separate from Method so that this information can be accessed without
        /// paying the initialization cost of loading the full Method.
        type MethodMeta = crate::MethodMeta;

        /// A facade class for loading programs and executing methods within them.
        #[namespace = "executorch::extension"]
        type Module;

        /// Enum to define loading behavior.
        type ModuleLoadMode = crate::ModuleLoadMode;

        /// A specification of `ArrayRef<EValue>`.
        type ArrayRefEValue = crate::ArrayRefEValue;

        /// A vector of `EValue`.
        type VecEValue = crate::VecEValue;

        /// Constructs an instance by loading a program from a file with specified
        /// memory locking behavior.
        ///
        /// # Arguments
        /// - `file_path`: The path to the ExecuTorch program file to load.
        /// - `load_mode`: The loading mode to use.
        #[namespace = "executorch_rs"]
        fn Module_new(
            file_path: &str,
            load_mode: ModuleLoadMode,
            // event_tracer: *mut cxx::UniquePtr<crate::executorch::runtime::EventTracer>,
        ) -> UniquePtr<Module>;

        /// Load the program if needed.
        ///
        /// # Arguments
        /// - `verification`: The type of verification to do before returning success.
        ///
        /// # Returns
        /// An Error to indicate success or failure of the loading process.
        #[namespace = "executorch_rs"]
        fn Module_load(self_: Pin<&mut Module>, verification: ProgramVerification) -> Error;

        /// Get a list of method names available in the loaded program.
        ///
        /// Loads the program and method if needed.
        ///
        /// # Arguments
        /// - `method_names_out`: A mutable reference to a vector that will be filled with the method names.
        ///
        /// # Returns
        /// A error indicating whether the method names retrieval was successful or not.
        ///
        /// # Safety
        /// The `method_names_out` vector must be valid for the lifetime of the function.
        /// The `method_names_out` vector can be used only if the function returns `Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_method_names(
            self_: Pin<&mut Module>,
            method_names_out: &mut Vec<String>,
        ) -> Error;

        /// Load a specific method from the program and set up memory management if
        /// needed.
        ///
        /// The loaded method is cached to reuse the next time it's executed.
        ///
        /// # Arguments
        /// - `method_name`: The name of the method to load.
        ///
        /// # Returns
        /// An Error to indicate success or failure.
        #[namespace = "executorch_rs"]
        fn Module_load_method(self_: Pin<&mut Module>, method_name: &str) -> Error;

        /// Checks if a specific method is loaded.
        ///
        /// # Arguments
        /// - `method_name`: The name of the method to check.
        ///
        /// # Returns
        /// `true` if the method specified by `method_name` is loaded, `false` otherwise.
        #[namespace = "executorch_rs"]
        fn Module_is_method_loaded(self_: &Module, method_name: &str) -> bool;

        /// Get a method metadata struct by method name.
        ///
        /// Loads the program and method if needed.
        ///
        /// # Arguments
        /// - `method_name`: The name of the method to get the metadata for.
        /// - `method_meta_out`: A mutable reference to a `MethodMeta` struct that will be filled with the metadata.
        ///
        /// # Returns
        /// A error indicating whether the metadata retrieval was successful or not.
        ///
        /// # Safety
        /// The `method_meta_out` struct must be valid for the lifetime of the function.
        /// The `method_meta_out` struct can be used only if the function returns `Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_method_meta(
            self_: Pin<&mut Module>,
            method_name: &str,
            method_meta_out: *mut MethodMeta,
        ) -> Error;

        /// Execute a specific method with the given input values and retrieve the
        /// output values. Loads the program and method before executing if needed.
        ///
        /// # Arguments
        /// - `method_name`: The name of the method to execute.
        /// - `inputs`: A vector of input values to be passed to the method.
        /// - `outputs`: A mutable reference to a vector that will be filled with the output values from the method.
        ///
        /// # Returns
        /// A error indicating whether the execution was successful or not.
        ///
        /// # Safety
        /// The `outputs` vector must be valid for the lifetime of the function.
        /// The `outputs` vector can be used only if the function returns `Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_execute(
            self_: Pin<&mut Module>,
            method_name: &str,
            inputs: ArrayRefEValue,
            outputs: *mut VecEValue,
        ) -> Error;
    }
}

unsafe impl ExternType for crate::ModuleLoadMode {
    type Id = type_id!("ModuleLoadMode");
    type Kind = cxx::kind::Trivial;
}
