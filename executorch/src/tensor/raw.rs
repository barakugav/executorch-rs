use std::marker::PhantomData;
use std::pin::Pin;

use crate::memory::{Storable, Storage};
use crate::tensor::{
    DimOrderType, Scalar, ScalarType, SizesType, StridesType, TensorAccessor, TensorAccessorInner,
    TensorAccessorMut,
};
use crate::util::{Destroy, IntoCpp, IntoRust, NonTriviallyMovable, __ArrayRefImpl, c_new};
use crate::{CError, Error, Result};
use executorch_sys as et_c;

/// A raw tensor that does not own the underlying data.
///
/// This struct is a low-level match to the C++ `Tensor` class. The tensor does not own its data,
/// but rather point to a [`RawTensorImpl`]. A [`Tensor`](crate::tensor::Tensor) or any of its
/// variants (`TensorAny`, `TensorMut`, etc) is preferred for most use cases, but this struct
/// is exposed for low level users who need to avoid code size overhead (avoiding the regular
/// tensor generics).
///
/// The struct does not enforce any mutability rules, and the caller must ensure that the tensor
/// is used correctly according to its mutability.
pub struct RawTensor<'a>(
    NonTriviallyMovable<'a, et_c::TensorStorage>,
    // phantom for the lifetime of the TensorImpl we depends on
    PhantomData<&'a ()>,
);
impl<'a> RawTensor<'a> {
    /// Create a new tensor in a boxed heap memory.
    ///
    /// # Safety
    ///
    /// The returned tensor will allow to mutate the underlying data (which is owned by the
    /// `RawTensorImpl`), so the caller must ensure that the tensor is used correctly according to
    /// its mutability.
    #[cfg(feature = "alloc")]
    pub unsafe fn new(tensor_impl: &'a RawTensorImpl) -> Self {
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
    /// The returned tensor will allow to mutate the underlying data (which is owned by the
    /// `RawTensorImpl`), so the caller must ensure that the tensor is used correctly according to
    /// its mutability.
    pub unsafe fn new_in_storage(
        tensor_impl: &'a RawTensorImpl,
        storage: Pin<&'a mut Storage<RawTensor<'_>>>,
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
    /// and that the tensor is compatible with the data generic.
    /// The created tensor should not be modified as we take an immutable reference to the given
    /// Cpp tensor reference.
    pub(crate) unsafe fn from_inner_ref(tensor: et_c::TensorRef) -> Self {
        debug_assert!(!tensor.ptr.is_null());
        let tensor = unsafe { &*(tensor.ptr as *const et_c::TensorStorage) };
        Self(NonTriviallyMovable::from_ref(tensor), PhantomData)
    }

    /// Create a new mutable tensor from a mutable Cpp reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given tensor is valid for the lifetime of the new tensor.
    #[allow(unused)]
    pub(crate) unsafe fn from_inner_ref_mut(tensor: et_c::TensorRefMut) -> Self {
        debug_assert!(!tensor.ptr.is_null());
        let tensor = unsafe { &mut *(tensor.ptr as *mut et_c::TensorStorage) };
        Self(NonTriviallyMovable::from_mut_ref(tensor), PhantomData)
    }

    /// Get the underlying Cpp tensor.
    pub(crate) fn as_cpp(&self) -> et_c::TensorRef {
        et_c::TensorRef {
            ptr: self.0.as_ref() as *const et_c::TensorStorage as *const _,
        }
    }

    /// Get a mutable reference to the underlying Cpp tensor.
    ///
    /// # Safety
    ///
    /// The caller can not move out of the returned mut reference, and should use this function only
    /// if the tensor was created with a mutable tensor impl.
    pub(crate) unsafe fn as_cpp_mut(&mut self) -> Option<et_c::TensorRefMut> {
        // Safety: the caller does not move out of the returned mut reference.
        Some(et_c::TensorRefMut {
            ptr: unsafe { self.0.as_mut()? } as *mut et_c::TensorStorage as *mut _,
        })
    }

    /// Returns the size of the tensor in bytes.
    ///
    /// NOTE: Only the alive space is returned not the total capacity of the
    /// underlying data blob.
    pub fn nbytes(&self) -> usize {
        unsafe { et_c::executorch_Tensor_nbytes(self.as_cpp()) }
    }

    /// Returns the size of the tensor at the given dimension.
    ///
    /// NOTE: that size() intentionally does not return SizeType even though it
    /// returns an element of an array of SizeType. This is to help make calls of
    /// this method more compatible with at::Tensor, and more consistent with the
    /// rest of the methods on this class and in ETensor.
    pub fn size(&self, dim: usize) -> usize {
        unsafe { et_c::executorch_Tensor_size(self.as_cpp(), dim) }
    }

    /// Returns the tensor's number of dimensions.
    pub fn dim(&self) -> usize {
        unsafe { et_c::executorch_Tensor_dim(self.as_cpp()) }
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        unsafe { et_c::executorch_Tensor_numel(self.as_cpp()) }
    }

    /// Returns the type of the elements in the tensor (int32, float, bool, etc).
    pub fn scalar_type(&self) -> ScalarType {
        unsafe { et_c::executorch_Tensor_scalar_type(self.as_cpp()) }.rs()
    }

    /// Returns the size in bytes of one element of the tensor.
    pub fn element_size(&self) -> usize {
        unsafe { et_c::executorch_Tensor_element_size(self.as_cpp()) }
    }

    /// Returns the sizes of the tensor at each dimension.
    pub fn sizes(&self) -> &[SizesType] {
        unsafe {
            let arr = et_c::executorch_Tensor_sizes(self.as_cpp());
            debug_assert!(!arr.data.is_null());
            std::slice::from_raw_parts(arr.data, arr.len)
        }
    }

    /// Returns the order the dimensions are laid out in memory.
    pub fn dim_order(&self) -> &[DimOrderType] {
        unsafe {
            let arr = et_c::executorch_Tensor_dim_order(self.as_cpp());
            debug_assert!(!arr.data.is_null());
            std::slice::from_raw_parts(arr.data, arr.len)
        }
    }

    /// Returns the strides of the tensor at each dimension.
    pub fn strides(&self) -> &[StridesType] {
        unsafe {
            let arr = et_c::executorch_Tensor_strides(self.as_cpp());
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
        let ptr = unsafe { et_c::executorch_Tensor_const_data_ptr(self.as_cpp()) };
        debug_assert!(!ptr.is_null());
        ptr as *const ()
    }

    /// Returns a mutable pointer to the underlying data blob.
    ///
    /// # Returns
    ///
    /// Returns a mutable pointer to the underlying data blob, and sometimes `None` if the tensor is not mutable.
    /// `Some` may be returned in cases the tensor itself is mutable (owned or created by a mutable reference),
    /// but the underlying data is not mutable (e.g. a tensor created from a immutable tensor impl).
    /// In such cases, the caller should not call this function.
    /// This is why this function is marked as `unsafe`.
    ///
    /// # Safety
    ///
    /// The caller must access the values in the returned pointer according to the type, sizes, dim order and strides
    /// of the tensor.
    /// The caller should call this function only if the tensor was created with a mutable tensor impl.
    pub fn as_mut_ptr_raw(&mut self) -> Option<*mut ()> {
        let tensor = unsafe { self.as_cpp_mut()? };
        let tensor = et_c::TensorRef { ptr: tensor.ptr };
        let ptr = unsafe { et_c::executorch_Tensor_mutable_data_ptr(tensor) };
        debug_assert!(!ptr.is_null());
        Some(ptr as *mut ())
    }

    fn coordinate_to_index(&self, coordinate: &[usize]) -> Option<usize> {
        let index = unsafe {
            et_c::executorch_Tensor_coordinate_to_index(
                self.as_cpp(),
                et_c::ArrayRefUsizeType::from_slice(coordinate),
            )
        };
        if index < 0 {
            None
        } else {
            Some(index as usize)
        }
    }
    unsafe fn coordinate_to_index_unchecked(&self, coordinate: &[usize]) -> usize {
        cfg_if::cfg_if! { if #[cfg(debug_assertions)] {
            let index = self.coordinate_to_index(coordinate);
            unsafe { index.unwrap_unchecked() }
        } else {
            let index = unsafe {
                et_c::executorch_Tensor_coordinate_to_index_unchecked(
                    self.as_cpp(),
                    et_c::ArrayRefUsizeType::from_slice(coordinate),
                )
            };
            index as usize
        } }
    }

    /// Get a reference to the element at `index`, without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_unchecked<S: Scalar>(&self, index: &[usize]) -> &S {
        let index = unsafe { self.coordinate_to_index_unchecked(index) };
        let base_ptr = self.as_ptr_raw() as *const S;
        debug_assert!(!base_ptr.is_null());
        unsafe { &*base_ptr.add(index) }
    }

    /// Get a mutable reference to the element at `index`, without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    /// The caller must ensure that the tensor was created with a mutable tensor impl.
    pub unsafe fn get_unchecked_mut<S: Scalar>(&mut self, index: &[usize]) -> &mut S {
        let index = unsafe { self.coordinate_to_index_unchecked(index) };
        let base_ptr = unsafe { self.as_mut_ptr_raw().unwrap_unchecked() } as *mut S;
        debug_assert!(!base_ptr.is_null());
        unsafe { &mut *base_ptr.add(index) }
    }

    /// Safety: the caller must ensure that type `S` is the correct scalar type of the tensor.
    pub(super) unsafe fn get_without_type_check<S: Scalar>(&self, index: &[usize]) -> Option<&S> {
        let index = self.coordinate_to_index(index)?;
        let base_ptr = self.as_ptr_raw() as *const S;
        debug_assert!(!base_ptr.is_null());
        Some(unsafe { &*base_ptr.add(index) })
    }

    /// Safety: the caller must ensure that type `S` is the correct scalar type of the tensor.
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to the element at `index`, or `None` if the index is out of bounds.
    /// The caller must ensure that the tensor was created with a mutable tensor impl.
    /// `Some` may be returned in cases the tensor itself is mutable (owned or created by a mutable reference),
    /// but the underlying data is not mutable (e.g. a tensor created from a immutable tensor impl).
    /// In such cases, the caller should not call this function.
    /// This is why this function is marked as `unsafe`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the type `S` is the correct scalar type of the tensor,
    /// and that the tensor was created with a mutable tensor impl.
    pub(super) unsafe fn get_without_type_check_mut<S: Scalar>(
        &mut self,
        index: &[usize],
    ) -> Option<&mut S> {
        let index = self.coordinate_to_index(index)?;
        let base_ptr = self.as_mut_ptr_raw()? as *mut S;
        debug_assert!(!base_ptr.is_null());
        Some(unsafe { &mut *base_ptr.add(index) })
    }

    /// Get a reference to the element at `index`, or `None` if the scalar type of the tensor does not
    /// match `S` or the index is out of bounds.
    pub fn get_as_typed<S: Scalar>(&self, index: &[usize]) -> Option<&S> {
        if self.scalar_type() == S::TYPE {
            // Safety: the scalar type is checked
            unsafe { self.get_without_type_check(index) }
        } else {
            None
        }
    }

    /// Get a mutable reference to the element at `index`, or `None` if the scalar type of the tensor does not
    /// match `S` or the index is out of bounds or the tensor is not mutable.
    ///
    /// `Some` may be returned in cases the tensor itself is mutable (owned or created by a mutable reference),
    /// but the underlying data is not mutable (e.g. a tensor created from a immutable tensor impl).
    ///
    /// # Safety
    ///
    /// The caller must ensure the tensor was created with a mutable tensor impl,
    pub unsafe fn get_as_typed_mut<S: Scalar>(&mut self, index: &[usize]) -> Option<&mut S> {
        if self.scalar_type() == S::TYPE {
            // Safety: the scalar type is checked
            unsafe { self.get_without_type_check_mut(index) }
        } else {
            None
        }
    }

    fn accessor_inner<S: Scalar, const N: usize>(&self) -> Option<TensorAccessorInner<'_, S, N>> {
        if self.scalar_type() != S::TYPE || self.dim() != N {
            return None;
        }
        if !self.dim_order().iter().map(|d| *d as usize).eq(0..N) {
            panic!("Non-default dim order is not supported for TensorAccessorInner");
        }
        let data = self.as_ptr_raw() as *const S;
        let accessor = unsafe { TensorAccessorInner::new(data, self.sizes(), self.strides()) };
        Some(accessor)
    }

    /// Get an immutable accessor for the tensor.
    ///
    /// An accessor is a utility struct, templated over the type of the tensor elements and the number
    /// of dimensions, which make it very efficient to access tensor elements by index.
    /// See the [`TensorAccessor`] for more details.
    ///
    /// # Returns
    ///
    /// Returns an accessor if the scalar type of the tensor matches `S` and the number of dimensions
    /// matches `N`, otherwise returns `None`.
    pub fn accessor<S: Scalar, const N: usize>(&self) -> Option<TensorAccessor<'_, S, N>> {
        Some(TensorAccessor(self.accessor_inner()?))
    }

    /// Get a mutable accessor for the tensor.
    ///
    /// An accessor is a utility struct, templated over the type of the tensor elements and the number
    /// of dimensions, which make it very efficient to access tensor elements by index.
    /// See the [`TensorAccessorMut`] for more details.
    ///
    /// # Returns
    ///
    /// Returns an accessor if the scalar type of the tensor matches `S` and the number of dimensions
    /// matches `N`, otherwise returns `None`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the tensor was created with a mutable tensor impl.
    pub unsafe fn accessor_mut<S: Scalar, const N: usize>(
        &mut self,
    ) -> Option<TensorAccessorMut<'_, S, N>> {
        Some(TensorAccessorMut(self.accessor_inner()?))
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
impl Storable for RawTensor<'_> {
    type __Storage = et_c::TensorStorage;
}

/// A raw tensor implementation.
///
/// This struct is a low-level match to the C++ `TensorImpl` class.
/// A [`TensorImpl`](crate::tensor::TensorImpl) or any of its
/// variants (`TensorImplAny`, `TensorImplMut`, etc) is preferred for most uses, but this struct
/// is exposed for low level users who need to avoid code size overhead (avoiding the regular
/// tensor generics).
/// The struct does not enforce any mutability rules, and the caller must ensure that the tensor
/// is used correctly according to its mutability.
pub struct RawTensorImpl<'a>(et_c::TensorImpl, PhantomData<&'a ()>);
impl<'a> RawTensorImpl<'a> {
    /// Create a new TensorImpl from a pointer to the data.
    ///
    /// # Errors
    ///
    /// Returns an error if dim order is invalid, or if it doesn't match the strides, or if the strides are not dense,
    /// i.e. if the strides are not the default strides of some permutation of the sizes.
    ///
    /// # Panics
    ///
    /// If the sizes, dim_order or strides slices are of different lengths.
    ///
    /// # Safety
    ///
    /// The caller must ensure elements in the data can be safely accessed according to the scalar type, sizes,
    /// dim order and strides of the tensor.
    /// The caller must ensure that the data is valid for the lifetime of the TensorImpl.
    pub unsafe fn from_ptr<S: Scalar>(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Result<Self> {
        let dim = sizes.len();
        assert_eq!(dim, dim_order.len());
        assert_eq!(dim, strides.len());

        let sizes = sizes.as_ptr();
        let dim_order = dim_order.as_ptr();
        let strides = strides.as_ptr();

        debug_assert!(!sizes.is_null());
        debug_assert!(!data.is_null());
        debug_assert!(!dim_order.is_null());
        debug_assert!(!strides.is_null());

        let valid_strides = unsafe {
            et_c::executorch_is_valid_dim_order_and_strides(dim, sizes, dim_order, strides)
        };
        if !valid_strides {
            return Err(Error::CError(CError::InvalidArgument));
        }

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
                )
            })
        };
        Ok(Self(impl_, PhantomData))
    }
}
