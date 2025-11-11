//! Tensor struct is an input or output tensor to an executorch program.
//!
//! The two core structs are [`TensorImplBase`] and [`TensorBase`]:
//! - [`TensorImplBase`] is a container for the tensor's data/shape/strides/etc, and a concrete tensor point to such
//!   container. It does not own the data, only holds reference to it.
//!   It is templated with immutable/mutable and typed/erased marker types, and usually not instantiated directly,
//!   rather through [`TensorImpl`] and [`TensorImplMut`] type aliases.
//! - [`TensorBase`] is a "base" class for all immutable/mutable/typed/type-erased tensors though generics of marker
//!   types. It points to a a tensor implementation and does not own it.
//!   Usually it is not instantiated directly, rather through type aliases:
//!   - [`Tensor`] typed immutable tensor.
//!   - [`TensorMut`] typed mutable tensor.
//!   - [`TensorAny`] type-erased immutable tensor.
//!
//! A [`ScalarType`] enum represents the possible scalar types of a typed-erased tensor, which can be converted into a
//! typed tensor and visa-versa using the
//! [`into_type_erased`](`TensorBase::into_type_erased`) and [`into_typed`](`TensorBase::into_typed`) methods.
//!
//! Both [`TensorImplBase`] and [`TensorBase`] are templated with a [`Data`] type, which is a trait that defines the
//! tensor data type. The [`DataMut`] and [`DataTyped`] traits are used to define mutable and typed data types,
//! extending the `Data` trait. The structs [`View`], [`ViewMut`], [`ViewAny`], and [`ViewMutAny`] are market types
//! that implement these traits, and a user should not implement them for their own types.
//!
//! The [`TensorPtr`] is a smart pointer to a tensor that also manage the lifetime of the [`TensorImpl`], the
//! underlying data buffer and the metadata (sizes/strides/etc arrays) of the tensor.
//! It has the most user-friendly API, but can not be used in `no_std` environments.
//!
//! In addition to all of the above safe tensors, the [`RawTensor`] and [`RawTensorImpl`] structs exists,
//! which match the Cpp `Tensor` and `TensorImpl` as close as possible. These struct do not enforce mutability
//! guarantees at all, and expose unsafe API, but may be useful to low level users who want to avoid code size
//! overhead (avoiding the regular tensor structs generics) of the safe API.

mod scalar;

mod layout;
pub use layout::*;

mod raw;
pub use raw::*;

mod safe;
pub use safe::*;

mod accessor;
pub use accessor::*;

#[cfg(feature = "ndarray")]
mod array;
#[cfg(feature = "ndarray")]
pub use array::*;

mod fmt;

use executorch_sys as sys;

/// A type that represents the sizes (dimensions) of a tensor.
pub type SizesType = sys::SizesType;
/// A type that represents the order of the dimensions of a tensor.
pub type DimOrderType = sys::DimOrderType;
/// A type that represents the strides of a tensor.
///
/// Strides are in units of the elements size, not in bytes.
pub type StridesType = sys::StridesType;

pub use scalar::{Scalar, ScalarType};

#[cfg(feature = "tensor-ptr")]
mod ptr;
#[cfg(feature = "tensor-ptr")]
pub use ptr::*;
