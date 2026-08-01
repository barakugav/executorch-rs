// Clippy doesnt detect the 'Safety' comments in the cxx bridge.
#![allow(clippy::missing_safety_doc)]
// The ET_-prefixed C bridge type names are not UpperCamelCase.
#![allow(non_camel_case_types)]

use cxx::{ExternType, type_id};

#[cxx::bridge]
pub(crate) mod ffi {

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        /// Cpp executorch error type.
        type ET_Error = crate::ET_Error;

        /// Types of validation that a `ET_Program` can do before parsing the data.
        type ET_ProgramVerification = crate::ET_ProgramVerification;

        /// Describes a method in an ExecuTorch program.
        ///
        /// The program used to create a ET_MethodMeta object must outlive the ET_MethodMeta.
        /// It is separate from ET_Method so that this information can be accessed without
        /// paying the initialization cost of loading the full ET_Method.
        type ET_MethodMeta = crate::ET_MethodMeta;

        /// A facade class for loading programs and executing methods within them.
        #[namespace = "executorch::extension"]
        type Module;

        /// Enum to define loading behavior.
        type ET_ModuleLoadMode = crate::ET_ModuleLoadMode;

        /// A specification of `ArrayRef<EValue>`.
        type ET_ArrayRefEValue = crate::ET_ArrayRefEValue;

        /// A vector of `EValue`.
        type ET_VecEValue = crate::ET_VecEValue;

        /// EventTracer is a class that users can inherit and implement to log/serialize/stream etc.
        #[namespace = "executorch::runtime"]
        type EventTracer;

        /// An allocator used to allocate objects for the runtime.
        type ET_MemoryAllocator = crate::ET_MemoryAllocator;

        /// Redefinition of the [`ET_HierarchicalAllocator`](crate::ET_HierarchicalAllocator).
        type ET_HierarchicalAllocator = crate::ET_HierarchicalAllocator;

        /// Maps backend IDs to their load-time options.
        type ET_LoadBackendOptionsMap = crate::ET_LoadBackendOptionsMap;

        /// Constructs an instance by loading a program from a file with specified
        /// memory locking behavior.
        ///
        /// # Arguments
        ///
        /// - `file_path`: The path to the ExecuTorch program file to load.
        /// - `data_files`: The path to one or more .ptd file/s.
        /// - `load_mode`: The loading mode to use.
        /// - `event_tracer`: An EventTracer used for tracking and logging events, or null if not needed.
        /// - `share_memory_arenas`: When true, all methods loaded by this Module share a single set of
        ///   memory-planned buffers.
        #[namespace = "executorch_rs"]
        fn Module_new(
            file_path: &CxxString,
            data_files: &[&str],
            load_mode: ET_ModuleLoadMode,
            event_tracer: UniquePtr<EventTracer>,
            memory_allocator: UniquePtr<ET_MemoryAllocator>,
            temp_allocator: UniquePtr<ET_MemoryAllocator>,
            share_memory_arenas: bool,
        ) -> UniquePtr<Module>;

        /// Load the program if needed, optionally with per-delegate load-time options.
        ///
        /// # Arguments
        ///
        /// - `backend_options`: Per-delegate load-time options, or null. When non-null the Module
        ///   deep-copies it into internal storage, so the caller may drop the map immediately after
        ///   this returns.
        /// - `verification`: The type of verification to do before returning success.
        ///
        /// # Returns
        ///
        /// An ET_Error to indicate success or failure of the loading process.
        ///
        /// # Safety
        ///
        /// `backend_options` must be null or point to a valid `ET_LoadBackendOptionsMap`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_load(
            self_: Pin<&mut Module>,
            backend_options: *const ET_LoadBackendOptionsMap,
            verification: ET_ProgramVerification,
        ) -> ET_Error;

        /// Returns the deep-copied LoadBackendOptionsMap most recently installed
        /// via `load(LoadBackendOptionsMap, ...)`.
        ///
        /// If `load(LoadBackendOptionsMap, ...)` has never been called, returns a
        /// default-constructed (empty, `size() == 0`) map.
        ///
        /// # Returns
        ///
        /// Const reference to the Module-owned LoadBackendOptionsMap.
        #[namespace = "executorch_rs"]
        fn Module_backend_options(self_: &Module) -> &ET_LoadBackendOptionsMap;

        /// Checks if the program is loaded.
        #[namespace = "executorch_rs"]
        fn Module_is_loaded(self_: &Module) -> bool;

        /// Get the number of methods available in the loaded program.
        ///
        /// # Safety
        ///
        /// The `method_num_out` is valid only if the function returns `ET_Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_num_methods(
            self_: Pin<&mut Module>,
            method_num_out: *mut usize,
        ) -> ET_Error;

        /// Get a list of method names available in the loaded program.
        ///
        /// Loads the program and method if needed.
        ///
        /// # Arguments
        ///
        /// - `method_names_out`: A pointer to a (non initialized) vector that will be created and filled with
        ///    the method names.
        ///
        /// # Returns
        ///
        /// A error indicating whether the method names retrieval was successful or not.
        ///
        /// # Safety
        ///
        /// The `method_names_out` vector can be used only if the function returns `ET_Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_method_names(
            self_: Pin<&mut Module>,
            method_names_out: *mut Vec<String>,
        ) -> ET_Error;

        /// Load a specific method from the program and set up memory management if
        /// needed.
        ///
        /// The loaded method is cached to reuse the next time it's executed.
        ///
        /// # Arguments
        ///
        /// - `method_name`: The name of the method to load.
        ///
        /// # Returns
        ///
        /// An ET_Error to indicate success or failure.
        #[namespace = "executorch_rs"]
        unsafe fn Module_load_method(
            self_: Pin<&mut Module>,
            method_name: &CxxString,
            planned_memory: *mut ET_HierarchicalAllocator,
            event_tracer: *mut EventTracer,
        ) -> ET_Error;

        /// Unload a specific method from the program.
        ///
        /// # Arguments
        /// - `method_name`: The name of the method to unload.
        ///
        /// # Returns
        ///
        /// True if the method is unloaded, false if no-op.
        #[namespace = "executorch_rs"]
        unsafe fn Module_unload_method(self_: Pin<&mut Module>, method_name: &CxxString) -> bool;

        /// Checks if a specific method is loaded.
        ///
        /// # Arguments
        ///
        /// - `method_name`: The name of the method to check.
        ///
        /// # Returns
        ///
        /// `true` if the method specified by `method_name` is loaded, `false` otherwise.
        #[namespace = "executorch_rs"]
        fn Module_is_method_loaded(self_: &Module, method_name: &CxxString) -> bool;

        /// Get a method metadata struct by method name.
        ///
        /// Loads the program if needed.
        ///
        /// # Arguments
        ///
        /// - `method_name`: The name of the method to get the metadata for.
        /// - `method_meta_out`: A mutable reference to a `ET_MethodMeta` struct that will be filled with the metadata.
        ///
        /// # Returns
        ///
        /// A error indicating whether the metadata retrieval was successful or not.
        ///
        /// # Safety
        ///
        /// The `method_meta_out` struct must be valid for the lifetime of the function.
        /// The `method_meta_out` struct can be used only if the function returns `ET_Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_method_meta(
            self_: Pin<&mut Module>,
            method_name: &CxxString,
            method_meta_out: *mut ET_MethodMeta,
        ) -> ET_Error;

        /// Execute a specific method with the given input values and retrieve the
        /// output values. Loads the program and method before executing if needed.
        ///
        /// # Arguments
        ///
        /// - `method_name`: The name of the method to execute.
        /// - `inputs`: A vector of input values to be passed to the method.
        /// - `outputs`: A mutable reference to a vector that will be filled with the output values from the method.
        ///
        /// # Returns
        ///
        /// A error indicating whether the execution was successful or not.
        ///
        /// # Safety
        ///
        /// The `outputs` vector must be valid for the lifetime of the function.
        /// The `outputs` vector can be used only if the function returns `ET_Error::Ok`.
        #[namespace = "executorch_rs"]
        unsafe fn Module_execute(
            self_: Pin<&mut Module>,
            method_name: &CxxString,
            inputs: ET_ArrayRefEValue,
            outputs: *mut ET_VecEValue,
        ) -> ET_Error;
    }
}

unsafe impl ExternType for crate::ET_HierarchicalAllocator {
    type Id = type_id!("ET_HierarchicalAllocator");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_ModuleLoadMode {
    type Id = type_id!("ET_ModuleLoadMode");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_LoadBackendOptionsMap {
    type Id = type_id!("ET_LoadBackendOptionsMap");
    type Kind = cxx::kind::Trivial;
}
