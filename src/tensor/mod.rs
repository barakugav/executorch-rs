//! Tensor struct is an input or output tensor to an executorch program.
//!
//! The two core structs are [`TensorImplBase`] and [`TensorBase`]:
//! - [`TensorImplBase`] is a container for the tensor's data/shape/strides/etc, and a concrete tensor point to such
//!     container. It does not own the data, only holds reference to it.
//!     It is templated with immutable/mutable and typed/erased marker types, and usually not instantiated directly,
//!     rather through [`TensorImpl`] and [`TensorImplMut`] type aliases.
//! - [`TensorBase`] is a "base" class for all immutable/mutable/typed/type-erased tensors though generics of marker
//!     types. It points to a a tensor implementation and does not own it.
//!     Usually it is not instantiated directly, rather through type aliases:
//!     - [`Tensor`] typed immutable tensor.
//!     - [`TensorMut`] typed mutable tensor.
//!     - [`TensorAny`] type-erased immutable tensor.
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

mod scalar;

#[cfg(feature = "ndarray")]
mod array;
#[cfg(feature = "ndarray")]
pub use array::*;

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::pin::Pin;

use crate::memory::{MemoryAllocator, Storable, Storage};
use crate::util::{Destroy, IntoCpp, IntoRust, NonTriviallyMovable, __ArrayRefImpl, c_new};
use crate::{CError, Error, Result};
use executorch_sys as et_c;

/// A type that represents the sizes (dimensions) of a tensor.
pub type SizesType = et_c::SizesType;
/// A type that represents the order of the dimensions of a tensor.
pub type DimOrderType = et_c::DimOrderType;
/// A type that represents the strides of a tensor.
pub type StridesType = et_c::StridesType;

pub use scalar::{Scalar, ScalarType};

/// A minimal Tensor type whose API is a source compatible subset of at::Tensor.
///
/// This class is a base class for all immutable/mutable/typed/type-erased tensors and is not meant to be
/// used directly.
/// Use the aliases such as [`Tensor`], [`TensorAny`] or [`TensorMut`] instead.
/// It is used to provide a common API for all of them.
pub struct TensorBase<'a, D: Data>(
    NonTriviallyMovable<'a, et_c::TensorStorage>,
    PhantomData<(
        // phantom for the lifetime of the TensorImpl we depends on
        &'a (),
        D,
    )>,
);
impl<'a, D: Data> TensorBase<'a, D> {
    /// Create a new tensor in a boxed heap memory.
    ///
    /// # Safety
    ///
    /// The caller must obtain a mutable reference to `tensor_impl` if the tensor is mutable.
    #[cfg(feature = "alloc")]
    unsafe fn new_boxed(tensor_impl: &'a TensorImplBase<D>) -> Self {
        let impl_ = &tensor_impl.0 as *const et_c::TensorImpl;
        let impl_ = impl_.cast_mut();
        // Safety: the closure init the pointer
        let tensor = unsafe {
            NonTriviallyMovable::new_boxed(|p: *mut et_c::TensorStorage| {
                let p = et_c::TensorRefMut { ptr: p as *mut _ };
                et_c::executorch_Tensor_new(p, impl_)
            })
        };
        Self(tensor, PhantomData)
    }

    /// Create a new tensor in the given storage.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new tensor is compatible with the given storage.
    unsafe fn new_in_storage_impl(
        tensor_impl: &'a TensorImplBase<D>,
        storage: Pin<&'a mut Storage<TensorBase<D>>>,
    ) -> Self {
        let impl_ = &tensor_impl.0 as *const et_c::TensorImpl;
        let impl_ = impl_.cast_mut();
        // Safety: the closure init the pointer
        let tensor = unsafe {
            NonTriviallyMovable::new_in_storage(
                |p: *mut executorch_sys::TensorStorage| {
                    let p = et_c::TensorRefMut { ptr: p as *mut _ };
                    et_c::executorch_Tensor_new(p, impl_)
                },
                storage,
            )
        };
        Self(tensor, PhantomData)
    }

    /// Create a new tensor from an immutable Cpp reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given tensor is valid for the lifetime of the new tensor,
    /// and that the tensor is compatible with the data generic. `D` must be immutable as we take immutable reference
    /// to the given tensor.
    pub(crate) unsafe fn from_inner_ref(tensor: et_c::TensorRef) -> Self {
        debug_assert!(!tensor.ptr.is_null());
        let tensor = unsafe { &*(tensor.ptr as *const et_c::TensorStorage) };
        Self(NonTriviallyMovable::from_ref(tensor), PhantomData)
    }

    /// Create a new mutable tensor from a mutable Cpp reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given tensor is valid for the lifetime of the new tensor,
    /// and that the tensor is compatible with the data generic.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_inner_ref_mut(tensor: et_c::TensorRefMut) -> Self {
        debug_assert!(!tensor.ptr.is_null());
        let tensor = unsafe { &mut *(tensor.ptr as *mut et_c::TensorStorage) };
        Self(NonTriviallyMovable::from_mut_ref(tensor), PhantomData)
    }

    /// Create a new tensor with the same internal data as the given tensor, but with different data generic.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new data generic is compatible with the data of the given tensor.
    pub(crate) unsafe fn convert_from<D2: Data>(tensor: TensorBase<'a, D2>) -> Self {
        Self(tensor.0, PhantomData)
    }

    /// Create a new tensor referencing the same internal data as the given tensor, but with different data generic.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new data generic is compatible with the data of the given tensor.
    /// `D2` must be immutable as we take immutable reference to the given tensor.
    pub(crate) unsafe fn convert_from_ref<D2: Data>(tensor: &'a TensorBase<D2>) -> Self {
        let inner = tensor.as_cpp_tensor();
        let inner = &*(inner.ptr as *const et_c::TensorStorage);
        Self(NonTriviallyMovable::from_ref(inner), PhantomData)
    }

    /// Create a new mutable tensor referencing the same internal data as the given tensor, but with different data
    /// generic.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new data generic is compatible with the data of the given tensor.
    pub(crate) unsafe fn convert_from_mut_ref<D2>(tensor: &'a mut TensorBase<D2>) -> Self
    where
        D2: DataMut,
    {
        // Safety: we are not moving out of the mut reference of the inner tensor
        let inner = unsafe { tensor.as_mut_cpp_tensor() };
        let inner = &mut *(inner.ptr as *mut et_c::TensorStorage);
        Self(NonTriviallyMovable::from_mut_ref(inner), PhantomData)
    }

    /// Get the underlying Cpp tensor.
    pub(crate) fn as_cpp_tensor(&self) -> et_c::TensorRef {
        et_c::TensorRef {
            ptr: self.0.as_ref() as *const et_c::TensorStorage as *const _,
        }
    }

    /// Get a mutable reference to the underlying Cpp tensor.
    ///
    /// # Safety
    ///
    /// The caller can not move out of the returned mut reference.
    pub(crate) unsafe fn as_mut_cpp_tensor(&mut self) -> et_c::TensorRefMut
    where
        D: DataMut,
    {
        // Safety: the caller does not move out of the returned mut reference.
        et_c::TensorRefMut {
            ptr: unsafe { self.0.as_mut() }.unwrap() as *mut et_c::TensorStorage as *mut _,
        }
    }

    /// Returns the size of the tensor in bytes.
    ///
    /// NOTE: Only the alive space is returned not the total capacity of the
    /// underlying data blob.
    pub fn nbytes(&self) -> usize {
        unsafe { et_c::executorch_Tensor_nbytes(self.as_cpp_tensor()) }
    }

    /// Returns the size of the tensor at the given dimension.
    ///
    /// NOTE: that size() intentionally does not return SizeType even though it
    /// returns an element of an array of SizeType. This is to help make calls of
    /// this method more compatible with at::Tensor, and more consistent with the
    /// rest of the methods on this class and in ETensor.
    pub fn size(&self, dim: usize) -> usize {
        unsafe { et_c::executorch_Tensor_size(self.as_cpp_tensor(), dim) }
    }

    /// Returns the tensor's number of dimensions.
    pub fn dim(&self) -> usize {
        unsafe { et_c::executorch_Tensor_dim(self.as_cpp_tensor()) }
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        unsafe { et_c::executorch_Tensor_numel(self.as_cpp_tensor()) }
    }

    /// Returns the type of the elements in the tensor (int32, float, bool, etc).
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { et_c::executorch_Tensor_scalar_type(self.as_cpp_tensor()) }.rs()
    }

    /// Returns the size in bytes of one element of the tensor.
    pub fn element_size(&self) -> usize {
        unsafe { et_c::executorch_Tensor_element_size(self.as_cpp_tensor()) }
    }

    /// Returns the sizes of the tensor at each dimension.
    pub fn sizes(&self) -> &[SizesType] {
        unsafe {
            let arr = et_c::executorch_Tensor_sizes(self.as_cpp_tensor());
            debug_assert!(!arr.data.is_null());
            std::slice::from_raw_parts(arr.data, arr.len)
        }
    }

    /// Returns the order the dimensions are laid out in memory.
    pub fn dim_order(&self) -> &[DimOrderType] {
        unsafe {
            let arr = et_c::executorch_Tensor_dim_order(self.as_cpp_tensor());
            debug_assert!(!arr.data.is_null());
            std::slice::from_raw_parts(arr.data, arr.len)
        }
    }

    /// Returns the strides of the tensor at each dimension.
    pub fn strides(&self) -> &[StridesType] {
        unsafe {
            let arr = et_c::executorch_Tensor_strides(self.as_cpp_tensor());
            debug_assert!(!arr.data.is_null());
            std::slice::from_raw_parts(arr.data, arr.len)
        }
    }

    /// Returns a pointer to the constant underlying data blob.
    ///
    /// # Safety
    ///
    /// The caller must access the values in the returned pointer according to the type, sizes, dim order and strides
    /// of the tensor.
    pub fn as_ptr_raw(&self) -> *const () {
        let ptr = unsafe { et_c::executorch_Tensor_const_data_ptr(self.as_cpp_tensor()) };
        debug_assert!(!ptr.is_null());
        ptr as *const ()
    }

    /// Converts this tensor into a type-erased tensor.
    pub fn into_type_erased(self) -> TensorBase<'a, D::TypeErased> {
        // Safety: D::TypeErased is compatible with D
        unsafe { TensorBase::<'a, D::TypeErased>::convert_from(self) }
    }

    /// Get a type erased tensor referencing the same internal data as this tensor.
    pub fn as_type_erased(&self) -> TensorBase<<D::Immutable as Data>::TypeErased> {
        // Safety: <D::Immutable as Data>::TypeErased is compatible with D and its immutable (we took &self)
        unsafe { TensorBase::<<D::Immutable as Data>::TypeErased>::convert_from_ref(self) }
    }

    /// Get a type erased mutable tensor referencing the same internal data as this tensor.
    pub fn as_type_erased_mut(&mut self) -> TensorBase<D::TypeErased>
    where
        D: DataMut,
    {
        // Safety: D::TypeErased is compatible with D
        unsafe { TensorBase::<D::TypeErased>::convert_from_mut_ref(self) }
    }

    /// Try to convert this tensor into a typed tensor with scalar type `S`.
    ///
    /// Fails if the scalar type of the tensor does not match the required one.
    pub fn try_into_typed<S: Scalar>(self) -> Result<TensorBase<'a, D::Typed<S>>, Self> {
        if self.scalar_type() != S::TYPE {
            return Err(self);
        }
        // Safety: the scalar type is checked, D::Typed is compatible with D
        Ok(unsafe { TensorBase::<'a, D::Typed<S>>::convert_from(self) })
    }

    /// Convert this tensor into a typed tensor with scalar type `S`.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the required one.
    #[track_caller]
    pub fn into_typed<S: Scalar>(self) -> TensorBase<'a, D::Typed<S>> {
        self.try_into_typed()
            .map_err(|_| Error::CError(CError::InvalidType))
            .unwrap()
    }

    /// Try to get a typed tensor with scalar type `S` referencing the same internal data as this tensor.
    ///
    /// Fails if the scalar type of the tensor does not match the required one.
    pub fn try_as_typed<S: Scalar>(&self) -> Option<TensorBase<<D::Immutable as Data>::Typed<S>>> {
        if self.scalar_type() != S::TYPE {
            return None;
        }
        // Safety: the scalar type is checked, <D::Immutable as Data>::Typed<S> is compatible with D and its
        //  immutable (we took &self)
        Some(unsafe { TensorBase::<<D::Immutable as Data>::Typed<S>>::convert_from_ref(self) })
    }

    /// Get a typed tensor with scalar type `S` referencing the same internal data as this tensor.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the required one.
    #[track_caller]
    pub fn as_typed<S: Scalar>(&self) -> TensorBase<<D::Immutable as Data>::Typed<S>> {
        self.try_as_typed()
            .ok_or(Error::CError(CError::InvalidType))
            .unwrap()
    }

    /// Try to get a mutable typed tensor with scalar type `S` referencing the same internal data as this tensor.
    ///
    /// Fails if the scalar type of the tensor does not match the required one.
    pub fn try_as_typed_mut<S: Scalar>(&mut self) -> Option<TensorBase<D::Typed<S>>>
    where
        D: DataMut,
    {
        if self.scalar_type() != S::TYPE {
            return None;
        }
        // Safety: the scalar type is checked, D::Typed<S> is compatible with D
        Some(unsafe { TensorBase::<D::Typed<S>>::convert_from_mut_ref(self) })
    }

    /// Get a mutable typed tensor with scalar type `S` referencing the same internal data as this tensor.
    ///
    /// # Panics
    ///
    /// If the scalar type of the tensor does not match the required one.
    #[track_caller]
    pub fn as_typed_mut<S: Scalar>(&mut self) -> TensorBase<D::Typed<S>>
    where
        D: DataMut,
    {
        self.try_as_typed_mut()
            .ok_or(Error::CError(CError::InvalidType))
            .unwrap()
    }

    fn coordinate_to_index(&self, coordinate: &[usize]) -> Option<usize> {
        let index = unsafe {
            et_c::executorch_Tensor_coordinate_to_index(
                self.as_cpp_tensor(),
                et_c::ArrayRefUsizeType::from_slice(coordinate),
            )
        };
        if index < 0 {
            None
        } else {
            Some(index as usize)
        }
    }

    /// Safety: the caller must ensure that type `S` is the correct scalar type of the tensor.
    unsafe fn get_without_type_check<S: Scalar>(&self, index: &[usize]) -> Option<&S> {
        let index = self.coordinate_to_index(index)?;
        let base_ptr = self.as_ptr_raw() as *const S;
        debug_assert!(!base_ptr.is_null());
        Some(unsafe { &*base_ptr.add(index) })
    }

    /// Safety: the caller must ensure that type `S` is the correct scalar type of the tensor.
    unsafe fn get_without_type_check_mut<S: Scalar>(&self, index: &[usize]) -> Option<&mut S>
    where
        D: DataMut,
    {
        let index = self.coordinate_to_index(index)?;
        let base_ptr = self.as_mut_ptr_raw() as *mut S;
        debug_assert!(!base_ptr.is_null());
        Some(unsafe { &mut *base_ptr.add(index) })
    }

    /// Get a reference to the element at `index`, or `None` if the scalar type of the tensor does not
    /// match `S` or the index is out of bounds.
    pub fn try_get<S: Scalar>(&self, index: &[usize]) -> Option<&S> {
        if self.scalar_type() == S::TYPE {
            // Safety: the scalar type is checked
            unsafe { self.get_without_type_check(index) }
        } else {
            None
        }
    }
}
impl Destroy for et_c::TensorStorage {
    unsafe fn destroy(&mut self) {
        unsafe {
            et_c::executorch_Tensor_destructor(et_c::TensorRefMut {
                ptr: self as *mut Self as *mut _,
            })
        }
    }
}
impl<D: Data> Storable for TensorBase<'_, D> {
    type __Storage = et_c::TensorStorage;
}

#[cfg(feature = "ndarray")]
impl<D: Data> std::fmt::Debug for TensorBase<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut st = f.debug_struct("Tensor");
        st.field("scalar_type", &self.scalar_type());

        fn add_data_field<S: Scalar + std::fmt::Debug, D: DataTyped<Scalar = S>>(
            this: TensorBase<D>,
            st: &mut std::fmt::DebugStruct,
        ) {
            match this.dim() {
                0 => st.field("data", &this.as_array::<ndarray::Ix0>()),
                1 => st.field("data", &this.as_array::<ndarray::Ix1>()),
                2 => st.field("data", &this.as_array::<ndarray::Ix2>()),
                3 => st.field("data", &this.as_array::<ndarray::Ix3>()),
                4 => st.field("data", &this.as_array::<ndarray::Ix4>()),
                5 => st.field("data", &this.as_array::<ndarray::Ix5>()),
                6 => st.field("data", &this.as_array::<ndarray::Ix6>()),
                _ => {
                    cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                        st.field("data", &this.as_array_dyn())
                    } else {
                        st.field("data", &"unsupported (too many dimensions)")
                    } }
                }
            };
        }
        fn add_data_field_unsupported(st: &mut std::fmt::DebugStruct) {
            st.field("data", &"unsupported");
        }

        match self.scalar_type() {
            ScalarType::Byte => add_data_field(self.as_typed::<u8>(), &mut st),
            ScalarType::Char => add_data_field(self.as_typed::<i8>(), &mut st),
            ScalarType::Short => add_data_field(self.as_typed::<i16>(), &mut st),
            ScalarType::Int => add_data_field(self.as_typed::<i32>(), &mut st),
            ScalarType::Long => add_data_field(self.as_typed::<i64>(), &mut st),
            ScalarType::Half => {
                add_data_field(self.as_typed::<crate::scalar::f16>(), &mut st);
            }
            ScalarType::Float => add_data_field(self.as_typed::<f32>(), &mut st),
            ScalarType::Double => add_data_field(self.as_typed::<f64>(), &mut st),
            ScalarType::ComplexHalf => {
                add_data_field(
                    self.as_typed::<crate::scalar::Complex<crate::scalar::f16>>(),
                    &mut st,
                );
            }
            ScalarType::ComplexFloat => {
                add_data_field(self.as_typed::<crate::scalar::Complex<f32>>(), &mut st);
            }
            ScalarType::ComplexDouble => {
                add_data_field(self.as_typed::<crate::scalar::Complex<f64>>(), &mut st);
            }
            ScalarType::Bool => add_data_field(self.as_typed::<bool>(), &mut st),
            ScalarType::QInt8 => add_data_field_unsupported(&mut st),
            ScalarType::QUInt8 => add_data_field_unsupported(&mut st),
            ScalarType::QInt32 => add_data_field_unsupported(&mut st),
            ScalarType::BFloat16 => {
                add_data_field(self.as_typed::<crate::scalar::bf16>(), &mut st);
            }
            ScalarType::QUInt4x2 => add_data_field_unsupported(&mut st),
            ScalarType::QUInt2x4 => add_data_field_unsupported(&mut st),
            ScalarType::Bits1x8 => add_data_field_unsupported(&mut st),
            ScalarType::Bits2x4 => add_data_field_unsupported(&mut st),
            ScalarType::Bits4x2 => add_data_field_unsupported(&mut st),
            ScalarType::Bits8 => add_data_field_unsupported(&mut st),
            ScalarType::Bits16 => add_data_field_unsupported(&mut st),
            ScalarType::Float8_e5m2 => add_data_field_unsupported(&mut st),
            ScalarType::Float8_e4m3fn => add_data_field_unsupported(&mut st),
            ScalarType::Float8_e5m2fnuz => add_data_field_unsupported(&mut st),
            ScalarType::Float8_e4m3fnuz => add_data_field_unsupported(&mut st),
            ScalarType::UInt16 => add_data_field(self.as_typed::<u16>(), &mut st),
            ScalarType::UInt32 => add_data_field(self.as_typed::<u32>(), &mut st),
            ScalarType::UInt64 => add_data_field(self.as_typed::<u64>(), &mut st),
        };
        st.finish()
    }
}

impl<D: DataTyped> TensorBase<'_, D> {
    /// Returns a pointer to the constant underlying data blob.
    pub fn as_ptr(&self) -> *const D::Scalar {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        self.as_ptr_raw() as *const D::Scalar
    }

    /// Get a reference to the element at `index`, or `None` if the index is out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&D::Scalar> {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        // Safety: the scalar type is checked
        unsafe { self.get_without_type_check(index) }
    }
}

impl<D: DataMut> TensorBase<'_, D> {
    /// Returns a mutable pointer to the underlying data blob.
    ///
    /// # Safety
    ///
    /// The caller must access the values in the returned pointer according to the type, sizes, dim order and strides
    /// of the tensor.
    pub fn as_mut_ptr_raw(&self) -> *mut () {
        let ptr = unsafe { et_c::executorch_Tensor_mutable_data_ptr(self.as_cpp_tensor()) };
        debug_assert!(!ptr.is_null());
        ptr as *mut ()
    }

    /// Get a mutable reference to the element at `index`, or `None` if the scalar type of the tensor does not
    /// match `S` or the index is out of bounds.
    pub fn try_get_mut<S: Scalar>(&mut self, index: &[usize]) -> Option<&mut S> {
        if self.scalar_type() == S::TYPE {
            // Safety: the scalar type is checked
            unsafe { self.get_without_type_check_mut(index) }
        } else {
            None
        }
    }
}
impl<D: DataTyped + DataMut> TensorBase<'_, D> {
    /// Returns a mutable pointer of type S to the underlying data blob.
    pub fn as_mut_ptr(&self) -> *mut D::Scalar {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        self.as_mut_ptr_raw().cast()
    }

    /// Get a mutable reference to the element at `index`, or `None` if the index is out of bounds.
    pub fn get_mut(&self, index: &[usize]) -> Option<&mut D::Scalar> {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        // Safety: the scalar type is checked
        unsafe { self.get_without_type_check_mut(index) }
    }
}

impl<D: DataTyped> Index<&[usize]> for TensorBase<'_, D> {
    type Output = D::Scalar;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let index = self
            .coordinate_to_index(index)
            .ok_or(Error::InvalidIndex)
            .unwrap();
        let base_ptr = self.as_ptr();
        debug_assert!(!base_ptr.is_null());
        unsafe { &*base_ptr.add(index) }
    }
}
impl<D: DataTyped + DataMut> IndexMut<&[usize]> for TensorBase<'_, D> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let index = self
            .coordinate_to_index(index)
            .ok_or(Error::InvalidIndex)
            .unwrap();
        let base_ptr = self.as_mut_ptr();
        unsafe { &mut *base_ptr.add(index) }
    }
}

/// A typed immutable tensor that does not own the underlying data.
pub type Tensor<'a, S> = TensorBase<'a, View<S>>;
impl<'a, S: Scalar> Tensor<'a, S> {
    /// Create a new [`Tensor`] from a [`TensorImpl`].
    ///
    /// The underlying Cpp object is allocated on the heap, which is preferred on systems in which allocations are
    /// available.
    /// For an identical version that allocates the object in a given storage (possibly on the stack), see the
    /// [`new_in_storage`][Tensor::new_in_storage] method.
    /// Note that the tensor data is not copied, and the required allocation is small.
    /// See [`Storage`] for more information.
    #[cfg(feature = "alloc")]
    pub fn new(tensor_impl: &'a TensorImpl<S>) -> Self {
        // Safety: both Self and TensorImpl are immutable
        unsafe { Self::new_boxed(tensor_impl) }
    }

    /// Create a new [`Tensor`] from a [`TensorImpl`] in the given storage.
    ///
    /// This function is identical to [`Tensor::new`][Tensor::new], but it allows to create the tensor without the
    /// use of a heap.
    /// Few examples of ways to create a tensor:
    /// ```rust,ignore
    /// // The tensor is allocated on the heap
    /// let tensor = Tensor::new(&tensor_impl);
    ///
    /// // The tensor is allocated on the stack
    /// let storage = executorch::storage!(Tensor<f32>);
    /// let tensor = Tensor::new_in_storage(&tensor_impl, storage);
    ///
    /// // The tensor is allocated using a memory allocator
    /// let allocator: impl AsRef<MemoryAllocator> = ...; // usually global
    /// let tensor = Tensor::new_in_storage(&tensor_impl, allocator.as_ref().allocate_pinned().unwrap());
    /// ```
    /// Note that the tensor data is not copied, and the required allocation is small.
    /// See [`Storage`] for more information.
    pub fn new_in_storage(
        tensor_impl: &'a TensorImpl<S>,
        storage: Pin<&'a mut Storage<Tensor<S>>>,
    ) -> Self {
        // Safety: both Self and TensorImpl are immutable
        unsafe { Self::new_in_storage_impl(tensor_impl, storage) }
    }

    /// Create a new [`Tensor`] from a [`TensorImpl`] in the given memory allocator.
    ///
    /// This function is identical to [`Tensor::new_in_storage`][Tensor::new_in_storage], but it allocates the storage
    /// using the given memory allocator.
    ///
    /// # Panics
    ///
    /// If the allocation fails.
    pub fn new_in_allocator(
        tensor_impl: &'a TensorImpl<S>,
        allocator: &'a MemoryAllocator<'a>,
    ) -> Self {
        let storage = allocator
            .allocate_pinned()
            .ok_or(Error::AllocationFailed)
            .unwrap();
        Self::new_in_storage(tensor_impl, storage)
    }
}

/// A typed mutable tensor that does not own the underlying data.
pub type TensorMut<'a, S> = TensorBase<'a, ViewMut<S>>;
impl<'a, S: Scalar> TensorMut<'a, S> {
    /// Create a new [`TensorMut`] from a [`TensorImplMut`].
    ///
    /// The underlying Cpp object is allocated on the heap, which is preferred on systems in which allocations are
    /// available.
    /// For an identical version that allocates the object in a given storage (possibly on the stack), see the
    /// [`new_in_storage`][TensorMut::new_in_storage] method.
    /// Note that the tensor data is not copied, and the required allocation is small.
    /// See [`Storage`] for more information.
    #[cfg(feature = "alloc")]
    pub fn new(tensor_impl: &'a mut TensorImplMut<S>) -> Self {
        // Safety: Self has a mutable data, and we indeed took a mutable reference to tensor_impl
        unsafe { Self::new_boxed(tensor_impl) }
    }

    /// Create a new [`TensorMut`] from a [`TensorImplMut`] in the given storage.
    ///
    /// This function is identical to  [`TensorMut::new`][TensorMut::new], but it allows to create the tensor without the
    /// use of a heap.
    /// Few examples of ways to create a tensor:
    /// ```rust,ignore
    /// // The tensor is allocated on the heap
    /// let tensor = TensorMut::new(&tensor_impl);
    ///
    /// // The tensor is allocated on the stack
    /// let storage = executorch::storage!(TensorMut<f32>);
    /// let tensor = TensorMut::new_in_storage(&tensor_impl, storage);
    ///
    /// // The tensor is allocated using a memory allocator
    /// let allocator: impl AsRef<MemoryAllocator> = ...; // usually global
    /// let tensor = TensorMut::new_in_storage(&tensor_impl, allocator.as_ref().allocate_pinned().unwrap());
    /// ```
    /// Note that the tensor data is not copied, and the required allocation is small.
    /// See the [`Storage`] struct for more information.
    pub fn new_in_storage(
        tensor_impl: &'a mut TensorImplMut<S>,
        storage: Pin<&'a mut Storage<TensorMut<S>>>,
    ) -> Self {
        // Safety: Self has a mutable data, and we indeed took a mutable reference to tensor_impl
        unsafe { Self::new_in_storage_impl(tensor_impl, storage) }
    }
}

/// A type-erased immutable tensor that does not own the underlying data.
pub type TensorAny<'a> = TensorBase<'a, ViewAny>;

/// A tensor implementation that does not own the underlying data.
///
/// This is a base class for [`TensorImpl`] and [`TensorImplMut`] and is not meant to be
/// used directly. It is used to provide a common API for both of them.
pub struct TensorImplBase<'a, D: Data>(et_c::TensorImpl, PhantomData<(&'a (), D)>);
impl<'a, D: Data> TensorImplBase<'a, D> {
    unsafe fn new<S: Scalar>(
        dim: usize,
        sizes: *const SizesType,
        data: *mut S,
        dim_order: *const DimOrderType,
        strides: *const StridesType,
    ) -> Self {
        debug_assert!(!sizes.is_null());
        debug_assert!(!data.is_null());
        debug_assert!(!dim_order.is_null());
        debug_assert!(!strides.is_null());
        let impl_ = unsafe {
            c_new(|this| {
                et_c::executorch_TensorImpl_new(
                    this,
                    S::TYPE.cpp(),
                    dim,
                    sizes as *mut SizesType,
                    data as *mut _,
                    dim_order as *mut DimOrderType,
                    strides as *mut StridesType,
                    et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
                );
            })
        };
        Self(impl_, PhantomData)
    }

    unsafe fn from_ptr_impl<S: Scalar>(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        let dim = sizes.len();
        assert_eq!(dim, dim_order.len());
        assert_eq!(dim, strides.len());
        unsafe {
            Self::new(
                dim,
                sizes.as_ptr(),
                data,
                dim_order.as_ptr(),
                strides.as_ptr(),
            )
        }
    }
}

/// An immutable tensor implementation that does not own the underlying data.
pub type TensorImpl<'a, S> = TensorImplBase<'a, View<S>>;
impl<'a, S: Scalar> TensorImpl<'a, S> {
    /// Create a new TensorImpl from a pointer to the data.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    ///     the tensor. The slice must be valid for the lifetime of the TensorImpl.
    /// * `data` - A pointer to the data of the tensor. The caller must ensure that the data is valid for the
    ///     lifetime of the TensorImpl.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Safety
    ///
    /// The caller must ensure elements in the data can be safely accessed according to the scalar type, sizes,
    /// dim_order and strides of the tensor.
    /// The caller must ensure that the data is valid for the lifetime of the TensorImpl.
    pub unsafe fn from_ptr(
        sizes: &'a [SizesType],
        data: *const S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        unsafe { Self::from_ptr_impl(sizes, data as *mut S, dim_order, strides) }
    }

    /// Create a new TensorImpl from a data slice.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    ///     the tensor. The slice must be valid for the lifetime of the TensorImpl.
    /// * `data` - The data of the tensor. The slice may be bigger than expected (according to the sizes and strides)
    ///     but not smaller.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    pub fn from_slice(
        sizes: &'a [SizesType],
        data: &'a [S],
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        // TODO: verify the data length make sense with the sizes/dim_order/strides
        let data_ptr = data.as_ptr() as *mut S;
        unsafe { Self::from_ptr_impl(sizes, data_ptr, dim_order, strides) }
    }
}

/// A mutable tensor implementation that does not own the underlying data.
pub type TensorImplMut<'a, S> = TensorImplBase<'a, ViewMut<S>>;
impl<'a, S: Scalar> TensorImplMut<'a, S> {
    /// Create a new TensorImplMut from a pointer to the data.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    ///     the tensor. The slice must be valid for the lifetime of the TensorImplMut.
    /// * `data` - A pointer to the data of the tensor. The caller must ensure that the data is valid for the
    ///     lifetime of the TensorImplMut, and that there is not more references to the data (as the passed pointer
    ///     will be used to mutate the data).
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Safety
    ///
    /// The caller must ensure elements in the data can be safely accessed and mutated according to the scalar type,
    /// sizes, dim_order and strides of the tensor.
    pub unsafe fn from_ptr(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        unsafe { Self::from_ptr_impl(sizes, data, dim_order, strides) }
    }

    ///  Create a new TensorImplMut from a data slice.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    ///    the tensor. The slice must be valid for the lifetime of the TensorImplMut.
    /// * `data` - The data of the tensor. The slice may be bigger than expected (according to the sizes and strides)
    ///   but not smaller.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    pub fn from_slice(
        sizes: &'a [SizesType],
        data: &'a mut [S],
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Self {
        // TODO: verify the data length make sense with the sizes/dim_order/strides
        let data_ptr = data.as_ptr() as *mut S;
        unsafe { Self::from_ptr_impl(sizes, data_ptr, dim_order, strides) }
    }
}

/// A marker trait that provide information about the data type of a [`TensorBase`] and [`TensorImplBase`]
pub trait Data {
    /// An immutable version of the data type.
    ///
    /// For example, if the data type is `ViewMut<f32>`, the immutable version is `View<f32>`.
    /// If the data is already immutable, the immutable version is the same as the data type.
    type Immutable: Data;

    /// A mutable version of the data type.
    ///
    /// For example, if the data type is `View<f32>`, the mutable version is `ViewMut<f32>`.
    /// If the data is already mutable, the mutable version is the same as the data type.
    type Mutable: DataMut;

    /// A type-erased version of the data type.
    ///
    /// For example, if the data type is `View<f32>`, the type-erased version is `ViewAny`.
    /// If the data is already type-erased, the type-erased version is the same as the data type.
    type TypeErased: Data;

    /// A typed version of the data type.
    ///
    /// For example, if the data type is `ViewAny`, the typed version is `View<S>`.
    /// If the data is already typed, the typed version is the same as the data type.
    type Typed<S: Scalar>: DataTyped<Scalar = S>;

    private_decl! {}
}
/// A marker trait extending [`Data`] that indicate that the data is mutable.
#[allow(dead_code)]
pub trait DataMut: Data {}
/// A marker trait extending [`Data`] that provide information about the scalar type of the data.
pub trait DataTyped: Data {
    /// The scalar type of the data.
    type Scalar: Scalar;
}

/// A marker type of typed immutable data of a tensor.
pub struct View<S: Scalar>(PhantomData<S>);
impl<S: Scalar> Data for View<S> {
    type Immutable = View<S>;
    type Mutable = ViewMut<S>;
    type TypeErased = ViewAny;
    type Typed<S2: Scalar> = View<S2>;
    private_impl! {}
}
impl<S: Scalar> DataTyped for View<S> {
    type Scalar = S;
}
/// A marker type of typed mutable data of a tensor.
pub struct ViewMut<S: Scalar>(PhantomData<S>);
impl<S: Scalar> Data for ViewMut<S> {
    type Immutable = View<S>;
    type Mutable = ViewMut<S>;
    type TypeErased = ViewMutAny;
    type Typed<S2: Scalar> = ViewMut<S2>;
    private_impl! {}
}
impl<S: Scalar> DataMut for ViewMut<S> {}
impl<S: Scalar> DataTyped for ViewMut<S> {
    type Scalar = S;
}

/// A marker type of type-erased immutable viewed data of a tensor.
pub struct ViewAny;
impl Data for ViewAny {
    type Immutable = ViewAny;
    type Mutable = ViewMutAny;
    type TypeErased = ViewAny;
    type Typed<S: Scalar> = View<S>;
    private_impl! {}
}

/// A marker type of type-erased mutable viewed data of a tensor.
pub struct ViewMutAny;
impl Data for ViewMutAny {
    type Immutable = ViewAny;
    type Mutable = ViewMutAny;
    type TypeErased = ViewMutAny;
    type Typed<S: Scalar> = ViewMut<S>;
    private_impl! {}
}
impl DataMut for ViewMutAny {}

#[cfg(feature = "tensor-ptr")]
mod ptr;
#[cfg(feature = "tensor-ptr")]
pub use ptr::*;

impl Storable for Option<TensorAny<'_>> {
    type __Storage = et_c::OptionalTensorStorage;
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::memory::BufferMemoryAllocator;
    use crate::storage;

    #[test]
    fn tensor_from_ptr() {
        for i in 0..3 {
            // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
            let sizes = [2, 3];
            let data = [1, 2, 3, 4, 5, 6];
            let dim_order = [0, 1];
            let strides = [3, 1];
            let tensor_impl =
                unsafe { TensorImpl::from_ptr(&sizes, data.as_ptr(), &dim_order, &strides) };

            let storage = storage!(Tensor<i32>);
            let mut allocator_buf = [0u8; 1024];
            let allocator = BufferMemoryAllocator::new(&mut allocator_buf);
            #[allow(unused_assignments)]
            let mut tensor = None;
            cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                if i == 0 {
                    tensor = Some(Tensor::new(&tensor_impl));
                } else if i == 1 {
                    tensor = Some(Tensor::new_in_storage(&tensor_impl, storage));
                } else {
                    tensor = Some(Tensor::new_in_allocator(&tensor_impl, &allocator));
                }
            } else {
                if i == 0 {
                    tensor = Some(Tensor::new_in_storage(&tensor_impl, storage));
                } else {
                    tensor = Some(Tensor::new_in_allocator(&tensor_impl, &allocator));
                }
            } }
            let tensor = tensor.unwrap();

            assert_eq!(tensor.nbytes(), 24);
            assert_eq!(tensor.size(0), 2);
            assert_eq!(tensor.size(1), 3);
            assert_eq!(tensor.dim(), 2);
            assert_eq!(tensor.numel(), 6);
            assert_eq!(tensor.scalar_type(), ScalarType::Int);
            assert_eq!(tensor.element_size(), 4);
            assert_eq!(tensor.sizes(), &[2, 3]);
            assert_eq!(tensor.dim_order(), &[0, 1]);
            assert_eq!(tensor.strides(), &[3, 1]);
            assert_eq!(tensor.as_ptr(), data.as_ptr());
        }
    }

    #[test]
    fn tensor_from_slice() {
        for i in 0..3 {
            // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
            let sizes = [2, 3];
            let data = [1, 2, 3, 4, 5, 6];
            let dim_order = [0, 1];
            let strides = [3, 1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides);

            let storage = storage!(Tensor<i32>);
            let mut allocator_buf = [0u8; 1024];
            let allocator = BufferMemoryAllocator::new(&mut allocator_buf);
            #[allow(unused_assignments)]
            let mut tensor = None;
            cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                if i == 0 {
                    tensor = Some(Tensor::new(&tensor_impl));
                } else if i == 1 {
                    tensor = Some(Tensor::new_in_storage(&tensor_impl, storage));
                } else {
                    tensor = Some(Tensor::new_in_allocator(&tensor_impl, &allocator));
                }
            } else {
                if i == 0 {
                    tensor = Some(Tensor::new_in_storage(&tensor_impl, storage));
                } else {
                    tensor = Some(Tensor::new_in_allocator(&tensor_impl, &allocator));
                }
            } }
            let tensor = tensor.unwrap();

            assert_eq!(tensor.nbytes(), 24);
            assert_eq!(tensor.size(0), 2);
            assert_eq!(tensor.size(1), 3);
            assert_eq!(tensor.dim(), 2);
            assert_eq!(tensor.numel(), 6);
            assert_eq!(tensor.scalar_type(), ScalarType::Int);
            assert_eq!(tensor.element_size(), 4);
            assert_eq!(tensor.sizes(), &[2, 3]);
            assert_eq!(tensor.dim_order(), &[0, 1]);
            assert_eq!(tensor.strides(), &[3, 1]);
            assert_eq!(tensor.as_ptr(), data.as_ptr());
        }
    }

    #[test]
    fn tensor_mut_from_ptr() {
        for _i in 0..2 {
            // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
            let sizes = [2, 3];
            let mut data = [1, 2, 3, 4, 5, 6];
            let data_ptr = data.as_ptr();
            let dim_order = [0, 1];
            let strides = [3, 1];
            let mut tensor_impl =
                unsafe { TensorImplMut::from_ptr(&sizes, data.as_mut_ptr(), &dim_order, &strides) };

            let storage = storage!(TensorMut<i32>);
            #[allow(unused_assignments)]
            let mut tensor = None;
            cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                if _i == 0 {
                    tensor = Some(TensorMut::new(&mut tensor_impl));
                } else {
                    tensor = Some(TensorMut::new_in_storage(&mut tensor_impl, storage));
                }
            } else {
                tensor = Some(TensorMut::new_in_storage(&mut tensor_impl, storage));
            } }
            let tensor = tensor.unwrap();

            assert_eq!(tensor.nbytes(), 24);
            assert_eq!(tensor.size(0), 2);
            assert_eq!(tensor.size(1), 3);
            assert_eq!(tensor.dim(), 2);
            assert_eq!(tensor.numel(), 6);
            assert_eq!(tensor.scalar_type(), ScalarType::Int);
            assert_eq!(tensor.element_size(), 4);
            assert_eq!(tensor.sizes(), &[2, 3]);
            assert_eq!(tensor.dim_order(), &[0, 1]);
            assert_eq!(tensor.strides(), &[3, 1]);
            assert_eq!(tensor.as_ptr(), data_ptr);
        }
    }

    #[test]
    fn tensor_mut_from_slice() {
        for _i in 0..2 {
            // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
            let sizes = [2, 3];
            let mut data = [1, 2, 3, 4, 5, 6];
            let data_ptr = data.as_ptr();
            let dim_order = [0, 1];
            let strides = [3, 1];
            let mut tensor_impl =
                TensorImplMut::from_slice(&sizes, &mut data, &dim_order, &strides);

            let storage = storage!(TensorMut<i32>);
            #[allow(unused_assignments)]
            let mut tensor = None;
            cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
                if _i == 0 {
                    tensor = Some(TensorMut::new(&mut tensor_impl));
                } else {
                    tensor = Some(TensorMut::new_in_storage(&mut tensor_impl, storage));
                }
            } else {
                tensor = Some(TensorMut::new_in_storage(&mut tensor_impl, storage));
            } }
            let tensor = tensor.unwrap();

            assert_eq!(tensor.nbytes(), 24);
            assert_eq!(tensor.size(0), 2);
            assert_eq!(tensor.size(1), 3);
            assert_eq!(tensor.dim(), 2);
            assert_eq!(tensor.numel(), 6);
            assert_eq!(tensor.scalar_type(), ScalarType::Int);
            assert_eq!(tensor.element_size(), 4);
            assert_eq!(tensor.sizes(), &[2, 3]);
            assert_eq!(tensor.dim_order(), &[0, 1]);
            assert_eq!(tensor.strides(), &[3, 1]);
            assert_eq!(tensor.as_ptr(), data_ptr);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn tensor_get_scalar_type() {
        fn test_scalar_type<S: Scalar>(data_allocator: impl FnOnce(usize) -> crate::alloc::Vec<S>) {
            let sizes = [2, 4, 17];
            let data = data_allocator(2 * 4 * 17);
            let dim_order = [0, 1, 2];
            let strides = [4 * 17, 17, 1];
            let tensor_impl =
                unsafe { TensorImpl::from_ptr(&sizes, data.as_ptr(), &dim_order, &strides) };
            let tensor = Tensor::new(&tensor_impl);
            assert_eq!(tensor.scalar_type(), S::TYPE);
        }

        test_scalar_type::<u8>(|size| vec![0; size]);
        test_scalar_type::<i8>(|size| vec![0; size]);
        test_scalar_type::<i16>(|size| vec![0; size]);
        test_scalar_type::<i32>(|size| vec![0; size]);
        test_scalar_type::<i64>(|size| vec![0; size]);
        test_scalar_type::<f32>(|size| vec![0.0; size]);
        test_scalar_type::<f64>(|size| vec![0.0; size]);
        test_scalar_type::<bool>(|size| vec![false; size]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn tensor_index() {
        let sizes = [4, 5, 3];
        let indices = (0..sizes[0] as usize)
            .flat_map(|x| (0..sizes[1] as usize).map(move |y| (x, y)))
            .flat_map(|(x, y)| (0..sizes[2] as usize).map(move |z| (x, y, z)));
        let data = indices
            .clone()
            .map(|(x, y, z)| x as i32 * 1337 - y as i32 * 87 + z as i32 * 13)
            .collect::<crate::alloc::Vec<_>>();
        let dim_order = [0, 1, 2];
        let strides = [15, 3, 1];
        let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides);
        let tensor = Tensor::new(&tensor_impl);

        assert!(tensor.get(&[4, 0, 0]).is_none());
        assert!(tensor.get(&[0, 5, 0]).is_none());
        assert!(tensor.get(&[0, 0, 3]).is_none());
        assert!(tensor.try_get::<i32>(&[4, 0, 0]).is_none());
        assert!(tensor.try_get::<i32>(&[0, 5, 0]).is_none());
        assert!(tensor.try_get::<i32>(&[0, 0, 3]).is_none());

        for (x, y, z) in indices.clone() {
            let actual1 = tensor[&[x, y, z]];
            let actual2 = tensor.get(&[x, y, z]).unwrap();
            let actual3 = tensor.try_get::<i32>(&[x, y, z]).unwrap();
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(actual1, expected);
            assert_eq!(*actual2, expected);
            assert_eq!(*actual3, expected);
        }

        let tensor = tensor.as_type_erased();
        for (x, y, z) in indices.clone() {
            let actual = tensor.try_get::<i32>(&[x, y, z]).unwrap();
            assert_eq!(*actual, x as i32 * 1337 - y as i32 * 87 + z as i32 * 13);
        }
        assert!(tensor.try_get::<f32>(&[0, 0, 0]).is_none())
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn tensor_index_mut() {
        let sizes = [4, 5, 3];
        let indices = (0..sizes[0] as usize)
            .flat_map(|x| (0..sizes[1] as usize).map(move |y| (x, y)))
            .flat_map(|(x, y)| (0..sizes[2] as usize).map(move |z| (x, y, z)));
        let mut data = indices.clone().map(|_| 0).collect::<crate::alloc::Vec<_>>();
        let dim_order = [0, 1, 2];
        let strides = [15, 3, 1];
        let mut tensor_impl = TensorImplMut::from_slice(&sizes, &mut data, &dim_order, &strides);
        let mut tensor = TensorMut::new(&mut tensor_impl);

        assert!(tensor.get_mut(&[4, 0, 0]).is_none());
        assert!(tensor.get_mut(&[0, 5, 0]).is_none());
        assert!(tensor.get_mut(&[0, 0, 3]).is_none());
        assert!(tensor.try_get_mut::<i32>(&[4, 0, 0]).is_none());
        assert!(tensor.try_get_mut::<i32>(&[0, 5, 0]).is_none());
        assert!(tensor.try_get_mut::<i32>(&[0, 0, 3]).is_none());

        // IndexMut
        for (x, y, z) in indices.clone() {
            assert_eq!(tensor[&[x, y, z]], 0);
        }
        for (x, y, z) in indices.clone() {
            tensor[&[x, y, z]] = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
        }
        for (x, y, z) in indices.clone() {
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(tensor[&[x, y, z]], expected);
        }
        for (x, y, z) in indices.clone() {
            tensor[&[x, y, z]] = 0;
        }

        // get_mut
        for (x, y, z) in indices.clone() {
            assert_eq!(tensor[&[x, y, z]], 0);
        }
        for (x, y, z) in indices.clone() {
            *tensor.get_mut(&[x, y, z]).unwrap() = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
        }
        for (x, y, z) in indices.clone() {
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(tensor[&[x, y, z]], expected);
        }
        for (x, y, z) in indices.clone() {
            *tensor.get_mut(&[x, y, z]).unwrap() = 0;
        }

        // try_get_mut
        for (x, y, z) in indices.clone() {
            assert_eq!(tensor[&[x, y, z]], 0);
        }
        for (x, y, z) in indices.clone() {
            *tensor.try_get_mut::<i32>(&[x, y, z]).unwrap() =
                x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
        }
        for (x, y, z) in indices.clone() {
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(tensor[&[x, y, z]], expected);
        }
        for (x, y, z) in indices.clone() {
            *tensor.try_get_mut::<i32>(&[x, y, z]).unwrap() = 0;
        }

        // try_get_mut of type-erased tensor
        let mut tensor = tensor.as_type_erased_mut();
        for (x, y, z) in indices.clone() {
            assert_eq!(*tensor.try_get_mut::<i32>(&[x, y, z]).unwrap(), 0);
        }
        for (x, y, z) in indices.clone() {
            *tensor.try_get_mut::<i32>(&[x, y, z]).unwrap() =
                x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
        }
        for (x, y, z) in indices.clone() {
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(*tensor.try_get_mut::<i32>(&[x, y, z]).unwrap(), expected);
        }
        for (x, y, z) in indices.clone() {
            *tensor.try_get_mut::<i32>(&[x, y, z]).unwrap() = 0;
        }
        assert!(tensor.try_get_mut::<f32>(&[0, 0, 0]).is_none())
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn into_type_erased() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor();
        let tensor = tensor.into_type_erased();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let _ = tensor.into_typed::<i32>();

        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let tensor = tensor_ptr.as_tensor_mut();
        let tensor = tensor.into_type_erased();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let tensor = tensor.into_typed::<i32>();
        // as_mut_ptr_raw is available only if the tensor is mutable
        assert!(!tensor.as_mut_ptr_raw().is_null());
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn as_type_erased() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor();
        let tensor = tensor.as_type_erased();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let _ = tensor.as_typed::<i32>();

        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut();
        let mut tensor = tensor.as_type_erased_mut();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let tensor = tensor.as_typed_mut::<i32>();
        // as_mut_ptr_raw is available only if the tensor is mutable
        assert!(!tensor.as_mut_ptr_raw().is_null());
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn try_into_typed() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor().into_type_erased();
        let tensor = tensor.try_into_typed::<f64>().map(|_| ()).unwrap_err();
        let tensor = tensor.try_into_typed::<i32>().map_err(|_| ()).unwrap();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);

        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let tensor = tensor_ptr.as_tensor_mut().into_type_erased();
        let tensor = tensor.try_into_typed::<f64>().map(|_| ()).unwrap_err();
        let tensor = tensor.try_into_typed::<i32>().map_err(|_| ()).unwrap();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn into_typed() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor().into_type_erased();
        let tensor = tensor.into_typed::<i32>();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);

        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let tensor = tensor_ptr.as_tensor_mut().into_type_erased();
        let tensor = tensor.into_typed::<i32>();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    #[should_panic]
    fn into_typed_wrong() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor().into_type_erased();
        let _ = tensor.into_typed::<f64>();
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn try_as_typed() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor().into_type_erased();
        assert!(tensor.try_as_typed::<f64>().is_none());
        let tensor = tensor.try_as_typed::<i32>().unwrap();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);

        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut().into_type_erased();
        assert!(tensor.try_as_typed_mut::<f64>().is_none());
        let tensor = tensor.try_as_typed_mut::<i32>().unwrap();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn as_typed() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor().into_type_erased();
        let tensor = tensor.as_typed::<i32>();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);

        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut().into_type_erased();
        let tensor = tensor.as_typed_mut::<i32>();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    #[should_panic]
    fn as_typed_wrong() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor().into_type_erased();
        let _ = tensor.as_typed::<f64>();
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    #[should_panic]
    fn as_typed_mut_wrong() {
        let mut tensor_ptr =
            TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4]).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut().into_type_erased();
        let _ = tensor.as_typed_mut::<f64>();
    }
}
