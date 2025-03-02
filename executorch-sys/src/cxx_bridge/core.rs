// Clippy doesnt detect the 'Safety' comments in the cxx bridge.
#![allow(clippy::missing_safety_doc)]

use cxx::{type_id, ExternType};

#[cxx::bridge]
pub(crate) mod ffi {

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        /// Redefinition of the [`MemoryAllocator`](crate::MemoryAllocator).
        type MemoryAllocator = crate::MemoryAllocator;

        /// Dynamically allocates memory using malloc() and frees all pointers at
        /// destruction time.
        ///
        /// For systems with malloc(), this can be easier than using a fixed-sized
        /// MemoryAllocator.
        #[namespace = "executorch::aten"]
        type MallocMemoryAllocator;

        /// Construct a new Malloc memory allocator.
        fn MallocMemoryAllocator_new() -> UniquePtr<MallocMemoryAllocator>;

        /// Get a pointer to the base class `MemoryAllocator`.
        ///
        /// Safety: The caller must ensure that the pointer is valid for the lifetime of the `MemoryAllocator`.
        unsafe fn MallocMemoryAllocator_as_memory_allocator(
            self_: Pin<&mut MallocMemoryAllocator>,
        ) -> *const MemoryAllocator;
    }
}

unsafe impl ExternType for crate::ScalarType {
    type Id = type_id!("ScalarType");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::TensorShapeDynamism {
    type Id = type_id!("TensorShapeDynamism");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::Error {
    type Id = type_id!("Error");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::MethodMeta {
    type Id = type_id!("MethodMeta");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ArrayRefEValue {
    type Id = type_id!("ArrayRefEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::VecEValue {
    type Id = type_id!("VecEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ProgramVerification {
    type Id = type_id!("ProgramVerification");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::MemoryAllocator {
    type Id = type_id!("MemoryAllocator");
    type Kind = cxx::kind::Trivial;
}
