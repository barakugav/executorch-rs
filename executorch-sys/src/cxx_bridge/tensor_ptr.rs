// Clippy doesnt detect the 'Safety' comments in the cxx bridge.
#![allow(clippy::missing_safety_doc)]
// TensorPtr_new mirrors the Cpp make_tensor_ptr signature, which takes 8 arguments.
#![allow(clippy::too_many_arguments)]
// The ET_-prefixed C bridge type names are not UpperCamelCase.
#![allow(non_camel_case_types)]

pub mod cxx_util {
    /// A wrapper around `std::any::Any` that can be used in a cxx bridge.
    ///
    /// This struct is useful to pass any Rust object to C++ code as `Box<RustAny>`, and the C++ code will call
    /// the destructor of the object when the `RustAny` object is dropped.
    pub struct RustAny {
        #[allow(unused)]
        inner: Box<dyn std::any::Any>,
    }
    impl RustAny {
        /// Create a new `RustAny` object.
        pub fn new(inner: Box<dyn std::any::Any>) -> Self {
            Self { inner }
        }
    }
}

use cxx_util::RustAny;

#[cxx::bridge]
pub(crate) mod ffi {

    extern "Rust" {
        #[namespace = "executorch_rs::cxx_util"]
        type RustAny;
    }

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        /// Redefinition of the [`ET_ScalarType`](crate::ET_ScalarType).
        type ET_ScalarType = crate::ET_ScalarType;
        /// Redefinition of the [`ET_TensorShapeDynamism`](crate::ET_TensorShapeDynamism).
        type ET_TensorShapeDynamism = crate::ET_TensorShapeDynamism;
        /// Redefinition of the [`ET_Device`](crate::ET_Device).
        type ET_Device = crate::ET_Device;
        /// A minimal Tensor type whose API is a source compatible subset of at::Tensor.
        #[namespace = "executorch::aten"]
        type Tensor;

        /// Create a new tensor pointer.
        ///
        /// The `device` parameter sets the Tensor's device location only — no data is allocated or
        /// copied. The caller is responsible for ensuring `data` already lives on the requested
        /// device. To copy CPU data to a device, use `TensorPtr_clone_to` instead.
        ///
        /// Arguments:
        /// - `sizes`: The dimensions of the tensor.
        /// - `data`: A pointer to the beginning of the data buffer.
        /// - `dim_order`: The order of the dimensions.
        /// - `strides`: The strides of the tensor, in units of elements (not bytes).
        /// - `scalar_type`: The scalar type of the tensor.
        /// - `dynamism`: The dynamism of the tensor.
        /// - `allocation`: A `Box<RustAny>` object that will be dropped when the tensor is dropped. Can be used to
        ///    manage the lifetime of the data buffer.
        /// - `device`: The device on which `data` resides.
        ///
        /// Returns a shared pointer to the tensor.
        ///
        /// # Safety
        ///
        /// The `data` pointer must be valid for the lifetime of the tensor, and accessing it according to the data
        /// type, sizes, dim order, and strides must be valid. The `data` pointer must reside on `device`.
        #[namespace = "executorch_rs"]
        unsafe fn TensorPtr_new(
            sizes: UniquePtr<CxxVector<i32>>,
            data: *mut u8,
            dim_order: UniquePtr<CxxVector<u8>>,
            strides: UniquePtr<CxxVector<i32>>,
            scalar_type: ET_ScalarType,
            dynamism: ET_TensorShapeDynamism,
            allocation: Box<RustAny>,
            device: ET_Device,
        ) -> SharedPtr<Tensor>;

        /// Creates a TensorPtr that manages a new Tensor with the same properties
        /// as the given Tensor, but with a copy of the data owned by the returned
        /// TensorPtr, or nullptr if the original data is null.
        ///
        /// Arguments:
        ///
        /// - `tensor`: The Tensor to clone.
        /// - `scalar_type`: The data type for the cloned tensor. The data will be
        ///   cast from the source tensor's type.
        ///
        /// Returns a new TensorPtr that manages a Tensor with the specified type
        /// and copied/cast data.
        #[namespace = "executorch_rs"]
        fn TensorPtr_clone(tensor: &Tensor, scalar_type: ET_ScalarType) -> SharedPtr<Tensor>;

        /// Clones a TensorPtr's data onto the given target device, allocating and copying as
        /// needed.
        ///
        /// The transfer direction is inferred from the source and target device: host-to-device
        /// when `target` is an accelerator, and device-to-host when `target` is CPU. Copies use the
        /// DeviceAllocator registered for the accelerator side; a device-backed result owns its
        /// memory and frees it via that allocator when destroyed.
        ///
        /// Source and target must differ in device domain: for a CPU-to-CPU copy use
        /// `TensorPtr_clone`, and device-to-device transfers are not supported.
        ///
        /// Arguments:
        ///
        /// - `tensor`: The source tensor whose data will be copied.
        /// - `device`: The destination device (CPU or an accelerator).
        ///
        /// Returns a TensorPtr backed by `device` memory containing the copied data.
        #[namespace = "executorch_rs"]
        fn TensorPtr_clone_to(tensor: SharedPtr<Tensor>, device: ET_Device) -> SharedPtr<Tensor>;
    }

    impl SharedPtr<Tensor> {}
}
