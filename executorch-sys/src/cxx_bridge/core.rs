// Clippy doesnt detect the 'Safety' comments in the cxx bridge.
#![allow(clippy::missing_safety_doc)]
// The ET_-prefixed C bridge type names are not UpperCamelCase.
#![allow(non_camel_case_types)]

use cxx::{type_id, ExternType};

#[cxx::bridge]
pub(crate) mod ffi {

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        /// Redefinition of the [`ET_MemoryAllocator`](crate::ET_MemoryAllocator).
        type ET_MemoryAllocator = crate::ET_MemoryAllocator;

        /// Convert a `ET_MemoryAllocator` into a `UniquePtr<ET_MemoryAllocator>`.
        ///
        /// The function moves the `ET_MemoryAllocator` into a `UniquePtr`, and calls the destructor of the original
        /// `ET_MemoryAllocator`. It does not free the object itself though.
        #[namespace = "executorch_rs"]
        fn BufferMemoryAllocator_into_memory_allocator_unique_ptr(
            self_: Pin<&mut ET_MemoryAllocator>,
        ) -> UniquePtr<ET_MemoryAllocator>;

        /// Dynamically allocates memory using malloc() and frees all pointers at
        /// destruction time.
        ///
        /// For systems with malloc(), this can be easier than using a fixed-sized
        /// ET_MemoryAllocator.
        #[namespace = "executorch::extension"]
        type MallocMemoryAllocator;

        /// Construct a new Malloc memory allocator.
        #[namespace = "executorch_rs"]
        fn MallocMemoryAllocator_new() -> UniquePtr<MallocMemoryAllocator>;

        /// Get a pointer to the base class `ET_MemoryAllocator`.
        ///
        /// Safety: The caller must ensure that the pointer is valid for the lifetime of the `ET_MemoryAllocator`.
        #[namespace = "executorch_rs"]
        unsafe fn MallocMemoryAllocator_as_memory_allocator(
            self_: Pin<&mut MallocMemoryAllocator>,
        ) -> *mut ET_MemoryAllocator;

        /// Convert a `UniquePtr<MallocMemoryAllocator>` into a `UniquePtr<ET_MemoryAllocator>`.
        #[namespace = "executorch_rs"]
        fn MallocMemoryAllocator_into_memory_allocator_unique_ptr(
            self_: UniquePtr<MallocMemoryAllocator>,
        ) -> UniquePtr<ET_MemoryAllocator>;

    }

    impl UniquePtr<ET_MemoryAllocator> {}
}

unsafe impl ExternType for crate::ET_ScalarType {
    type Id = type_id!("ET_ScalarType");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_TensorShapeDynamism {
    type Id = type_id!("ET_TensorShapeDynamism");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_Error {
    type Id = type_id!("ET_Error");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_MethodMeta {
    type Id = type_id!("ET_MethodMeta");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_ArrayRefEValue {
    type Id = type_id!("ET_ArrayRefEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_VecEValue {
    type Id = type_id!("ET_VecEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_ProgramVerification {
    type Id = type_id!("ET_ProgramVerification");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ET_MemoryAllocator {
    type Id = type_id!("ET_MemoryAllocator");
    type Kind = cxx::kind::Opaque;
}
