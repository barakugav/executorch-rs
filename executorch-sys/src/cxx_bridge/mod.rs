mod core;
pub use core::ffi::{
    BufferMemoryAllocator_into_memory_allocator_unique_ptr, MallocMemoryAllocator,
    MallocMemoryAllocator_as_memory_allocator,
    MallocMemoryAllocator_into_memory_allocator_unique_ptr, MallocMemoryAllocator_new,
};

#[cfg(feature = "module")]
mod module;
#[cfg(feature = "module")]
pub use module::ffi::{
    EventTracer, Module, Module_execute, Module_is_loaded, Module_is_method_loaded, Module_load,
    Module_load_method, Module_method_meta, Module_method_names, Module_new, Module_num_methods,
    Module_unload_method,
};

#[cfg(feature = "tensor-ptr")]
pub(crate) mod tensor_ptr;
#[cfg(feature = "tensor-ptr")]
pub use tensor_ptr::ffi::{Tensor, TensorPtr_clone, TensorPtr_new};
