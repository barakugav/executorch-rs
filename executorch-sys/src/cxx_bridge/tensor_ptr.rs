/// Utility functions and structs used by the cxx bridge.
pub mod cxx_util {
    /// A wrapper around `std::any::Any` that can be used in the cxx bridge.
    ///
    /// This struct is useful to pass any Rust object to C++ code as `Box<RustAny>`, and the C++ code will call
    /// the destructor of the object when the `RustAny` object is dropped.
    pub struct RustAny {
        #[allow(dead_code)]
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
pub mod ffi {

    extern "Rust" {
        #[namespace = "executorch_rs::cxx_util"]
        type RustAny;
    }

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        /// Redifinition of the [`ScalarType`](crate::executorch::runtime::etensor::ScalarType).
        #[namespace = "executorch::aten"]
        type ScalarType = crate::executorch::runtime::etensor::ScalarType;
        /// Redifinition of the [`TensorShapeDynamism`](crate::executorch::runtime::TensorShapeDynamism).
        #[namespace = "executorch::aten"]
        type TensorShapeDynamism = crate::executorch::runtime::TensorShapeDynamism;
        /// Redifinition of the [`Tensor`](crate::executorch::runtime::etensor::Tensor).
        #[namespace = "executorch_rs"]
        type Tensor = crate::executorch_rs::Tensor;

        /// Create a new tensor pointer.
        ///
        /// Arguments:
        /// - `sizes`: The dimensions of the tensor.
        /// - `data`: A pointer to the beginning of the data buffer.
        /// - `dim_order`: The order of the dimensions.
        /// - `strides`: The strides of the tensor.
        /// - `scalar_type`: The scalar type of the tensor.
        /// - `dynamism`: The dynamism of the tensor.
        /// - `allocation`: A `Box<RustAny>` object that will be dropped when the tensor is dropped. Can be used to
        ///    manage the lifetime of the data buffer.
        ///
        /// Returns a shared pointer to the tensor.
        ///
        /// # Safety
        ///
        /// The `data` pointer must be valid for the lifetime of the tensor, and accessing it according to the data
        /// type, sizes, dim order, and strides must be valid.
        #[namespace = "executorch_rs"]
        unsafe fn TensorPtr_new(
            sizes: UniquePtr<CxxVector<i32>>,
            data: *mut u8,
            dim_order: UniquePtr<CxxVector<u8>>,
            strides: UniquePtr<CxxVector<i32>>,
            scalar_type: ScalarType,
            dynamism: TensorShapeDynamism,
            allocation: Box<RustAny>,
        ) -> SharedPtr<Tensor>;
    }

    impl SharedPtr<Tensor> {}
}
