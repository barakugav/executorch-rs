use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::pin::Pin;

use super::{DimOrderType, RawTensor, RawTensorImpl, Scalar, ScalarType, SizesType, StridesType};
use crate::memory::{MemoryAllocator, Storable, Storage};
use crate::{CError, Error, Result};
use executorch_sys as et_c;

/// A minimal Tensor type whose API is a source compatible subset of at::Tensor.
///
/// This class is a base class for all immutable/mutable/typed/type-erased tensors and is not meant to be
/// used directly.
/// Use the aliases such as [`Tensor`], [`TensorAny`] or [`TensorMut`] instead.
/// It is used to provide a common API for all of them.
pub struct TensorBase<'a, D: Data>(pub(crate) RawTensor<'a>, PhantomData<D>);
impl<'a, D: Data> TensorBase<'a, D> {
    /// Create a new tensor in a boxed heap memory.
    ///
    /// # Safety
    ///
    /// The caller must obtain a mutable reference to `tensor_impl` if the tensor is mutable.
    #[cfg(feature = "alloc")]
    unsafe fn new_boxed(tensor_impl: &'a TensorImplBase<D>) -> Self {
        Self(RawTensor::new(&tensor_impl.0), PhantomData)
    }

    /// Create a new tensor in the given storage.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new tensor is compatible with the given storage.
    unsafe fn new_in_storage_impl(
        tensor_impl: &'a TensorImplBase<D>,
        storage: Pin<&'a mut Storage<TensorBase<'_, D>>>,
    ) -> Self {
        // Safety: the storage is identical
        let storage = unsafe {
            std::mem::transmute::<
                Pin<&'a mut Storage<TensorBase<'_, D>>>,
                Pin<&'a mut Storage<RawTensor<'_>>>,
            >(storage)
        };
        let tensor = RawTensor::new_in_storage(&tensor_impl.0, storage);
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
        Self(RawTensor::from_inner_ref(tensor), PhantomData)
    }

    /// Create a new mutable tensor from a mutable Cpp reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given tensor is valid for the lifetime of the new tensor,
    /// and that the tensor is compatible with the data generic.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_inner_ref_mut(tensor: et_c::TensorRefMut) -> Self {
        Self(RawTensor::from_inner_ref_mut(tensor), PhantomData)
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
        Self::from_inner_ref(tensor.as_cpp())
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
        let inner = unsafe { tensor.as_cpp_mut() };
        Self::from_inner_ref_mut(inner)
    }

    /// Get the underlying Cpp tensor.
    pub(crate) fn as_cpp(&self) -> et_c::TensorRef {
        self.0.as_cpp()
    }

    /// Get a mutable reference to the underlying Cpp tensor.
    ///
    /// # Safety
    ///
    /// The caller can not move out of the returned mut reference.
    pub(crate) unsafe fn as_cpp_mut(&mut self) -> et_c::TensorRefMut
    where
        D: DataMut,
    {
        let tensor = self.0.as_cpp_mut();
        unsafe { tensor.unwrap_unchecked() }
    }

    /// Returns the size of the tensor in bytes.
    ///
    /// NOTE: Only the alive space is returned not the total capacity of the
    /// underlying data blob.
    pub fn nbytes(&self) -> usize {
        self.0.nbytes()
    }

    /// Returns the size of the tensor at the given dimension.
    ///
    /// NOTE: that size() intentionally does not return SizeType even though it
    /// returns an element of an array of SizeType. This is to help make calls of
    /// this method more compatible with at::Tensor, and more consistent with the
    /// rest of the methods on this class and in ETensor.
    pub fn size(&self, dim: usize) -> usize {
        self.0.size(dim)
    }

    /// Returns the tensor's number of dimensions.
    pub fn dim(&self) -> usize {
        self.0.dim()
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.0.numel()
    }

    /// Returns the type of the elements in the tensor (int32, float, bool, etc).
    pub fn scalar_type(&self) -> ScalarType {
        self.0.scalar_type()
    }

    /// Returns the size in bytes of one element of the tensor.
    pub fn element_size(&self) -> usize {
        self.0.element_size()
    }

    /// Returns the sizes of the tensor at each dimension.
    pub fn sizes(&self) -> &[SizesType] {
        self.0.sizes()
    }

    /// Returns the order the dimensions are laid out in memory.
    pub fn dim_order(&self) -> &[DimOrderType] {
        self.0.dim_order()
    }

    /// Returns the strides of the tensor at each dimension.
    pub fn strides(&self) -> &[StridesType] {
        self.0.strides()
    }

    /// Returns a pointer to the constant underlying data blob.
    ///
    /// # Safety
    ///
    /// The caller must access the values in the returned pointer according to the type, sizes, dim order and strides
    /// of the tensor.
    pub fn as_ptr_raw(&self) -> *const () {
        self.0.as_ptr_raw()
    }

    /// Returns a pointer to the constant underlying data blob.
    pub fn as_ptr(&self) -> *const D::Scalar
    where
        D: DataTyped,
    {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        self.as_ptr_raw() as *const D::Scalar
    }
    /// Returns a mutable pointer to the underlying data blob.
    ///
    /// # Safety
    ///
    /// The caller must access the values in the returned pointer according to the type, sizes, dim order and strides
    /// of the tensor.
    pub fn as_mut_ptr_raw(&mut self) -> *mut ()
    where
        D: DataMut,
    {
        let ptr = self.0.as_mut_ptr_raw();
        unsafe { ptr.unwrap_unchecked() }
    }

    /// Returns a mutable pointer of type S to the underlying data blob.
    pub fn as_mut_ptr(&mut self) -> *mut D::Scalar
    where
        D: DataTyped + DataMut,
    {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        self.as_mut_ptr_raw().cast()
    }

    /// Get a reference to the element at `index`, or `None` if the index is out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&D::Scalar>
    where
        D: DataTyped,
    {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        // Safety: the scalar type is checked
        unsafe { self.0.get_without_type_check(index) }
    }

    /// Get a mutable reference to the element at `index`, or `None` if the index is out of bounds.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut D::Scalar>
    where
        D: DataTyped + DataMut,
    {
        debug_assert_eq!(self.scalar_type(), D::Scalar::TYPE);
        // Safety: the scalar type is checked
        unsafe { self.0.get_without_type_check_mut(index) }
    }

    /// Get a reference to the element at `index`, or `None` if the scalar type of the tensor does not
    /// match `S` or the index is out of bounds.
    pub fn get_as_typed<S: Scalar>(&self, index: &[usize]) -> Option<&S> {
        self.0.get_as_typed(index)
    }

    /// Get a mutable reference to the element at `index`, or `None` if the scalar type of the tensor does not
    /// match `S` or the index is out of bounds.
    pub fn get_as_typed_mut<S: Scalar>(&mut self, index: &[usize]) -> Option<&mut S>
    where
        D: DataMut,
    {
        unsafe { self.0.get_as_typed_mut(index) }
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
}
impl<D: Data> Storable for TensorBase<'_, D> {
    type __Storage = et_c::TensorStorage;
}

impl<D: DataTyped> Index<&[usize]> for TensorBase<'_, D> {
    type Output = D::Scalar;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let value = unsafe { self.0.get_without_type_check::<D::Scalar>(index) };
        value.ok_or(Error::InvalidIndex).unwrap()
    }
}
impl<D: DataTyped + DataMut> IndexMut<&[usize]> for TensorBase<'_, D> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let value = unsafe { self.0.get_without_type_check_mut::<D::Scalar>(index) };
        value.ok_or(Error::InvalidIndex).unwrap()
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
            .ok_or(Error::CError(CError::MemoryAllocationFailed))
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
pub struct TensorImplBase<'a, D: Data>(RawTensorImpl<'a>, PhantomData<D>);
impl<'a, D: Data> TensorImplBase<'a, D> {
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
    unsafe fn from_ptr_impl<S: Scalar>(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Result<Self> {
        let impl_ = RawTensorImpl::from_ptr(sizes, data, dim_order, strides)?;
        Ok(Self(impl_, PhantomData))
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
    ///   the tensor. The slice must be valid for the lifetime of the TensorImpl.
    /// * `data` - A pointer to the data of the tensor. The caller must ensure that the data is valid for the
    ///   lifetime of the TensorImpl.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Errors
    ///
    /// Returns an error if dim order is invalid, or if it doesn't match the strides, or if the strides are not dense,
    /// i.e. if the strides are not the default strides of some permutation of the sizes.
    ///
    /// # Panics
    ///
    /// If the data pointer is null or if the sizes, dim_order or strides slices are of different lengths.
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
    ) -> Result<Self> {
        unsafe { Self::from_ptr_impl(sizes, data as *mut S, dim_order, strides) }
    }

    /// Create a new TensorImpl from a data slice.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    ///   the tensor. The slice must be valid for the lifetime of the TensorImpl.
    /// * `data` - The data of the tensor. The slice may be bigger than expected (according to the sizes and strides)
    ///   but not smaller.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Errors
    ///
    /// Returns an error if dim order is invalid, or if it doesn't match the strides, or if the strides are not dense,
    /// i.e. if the strides are not the default strides of some permutation of the sizes.
    ///
    /// # Panics
    ///
    /// If the sizes, dim_order or strides slices are of different lengths.
    pub fn from_slice(
        sizes: &'a [SizesType],
        data: &'a [S],
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Result<Self> {
        // TODO: verify the data length make sense with the sizes/dim_order/strides
        let data_ptr = data.as_ptr() as *mut S;
        unsafe { Self::from_ptr_impl(sizes, data_ptr, dim_order, strides) }
    }

    /// Create a new TensorImpl from a scalar.
    ///
    /// The created tensor will be zero-dimensional.
    pub fn from_scalar(scalar: &'a S) -> Self {
        unsafe { Self::from_ptr(&[], scalar as *const S, &[], &[]).unwrap() }
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
    ///   the tensor. The slice must be valid for the lifetime of the TensorImplMut.
    /// * `data` - A pointer to the data of the tensor. The caller must ensure that the data is valid for the
    ///   lifetime of the TensorImplMut, and that there is not more references to the data (as the passed pointer
    ///   will be used to mutate the data).
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
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
    /// The caller must ensure elements in the data can be safely accessed and mutated according to the scalar type,
    /// sizes, dim_order and strides of the tensor.
    pub unsafe fn from_ptr(
        sizes: &'a [SizesType],
        data: *mut S,
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Result<Self> {
        unsafe { Self::from_ptr_impl(sizes, data, dim_order, strides) }
    }

    ///  Create a new TensorImplMut from a data slice.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes (dimensions) of the tensor. The length of this slice is the number of dimensions of
    ///   the tensor. The slice must be valid for the lifetime of the TensorImplMut.
    /// * `data` - The data of the tensor. The slice may be bigger than expected (according to the sizes and strides)
    ///   but not smaller.
    /// * `dim_order` - The order of the dimensions of the tensor, must have the same length as `sizes`.
    /// * `strides` - The strides of the tensor, must have the same length as `sizes`.
    ///
    /// # Errors
    ///
    /// Returns an error if dim order is invalid, or if it doesn't match the strides, or if the strides are not dense,
    /// i.e. if the strides are not the default strides of some permutation of the sizes.
    ///
    /// # Panics
    ///
    /// If the sizes, dim_order or strides slices are of different lengths.
    pub fn from_slice(
        sizes: &'a [SizesType],
        data: &'a mut [S],
        dim_order: &'a [DimOrderType],
        strides: &'a [StridesType],
    ) -> Result<Self> {
        // TODO: verify the data length make sense with the sizes/dim_order/strides
        let data_ptr = data.as_ptr() as *mut S;
        unsafe { Self::from_ptr_impl(sizes, data_ptr, dim_order, strides) }
    }

    /// Create a new TensorImplMut from a scalar.
    ///
    /// The created tensor will be zero-dimensional.
    pub fn from_scalar(scalar: &'a mut S) -> Self {
        unsafe { Self::from_ptr(&[], scalar as *mut S, &[], &[]).unwrap() }
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

impl Storable for Option<TensorAny<'_>> {
    type __Storage = et_c::OptionalTensorStorage;
}

#[cfg(test)]
mod tests {
    use crate::memory::BufferMemoryAllocator;
    use crate::storage;
    #[allow(unused_imports)]
    use crate::tensor::*;

    #[test]
    fn tensor_from_ptr() {
        for i in 0..3 {
            // Create a tensor with sizes [2, 3] and data [1, 2, 3, 4, 5, 6]
            let sizes = [2, 3];
            let data = [1, 2, 3, 4, 5, 6];
            let dim_order = [0, 1];
            let strides = [3, 1];
            let tensor_impl = unsafe {
                TensorImpl::from_ptr(&sizes, data.as_ptr(), &dim_order, &strides).unwrap()
            };

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
            let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();

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
            let mut tensor_impl = unsafe {
                TensorImplMut::from_ptr(&sizes, data.as_mut_ptr(), &dim_order, &strides).unwrap()
            };

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
                TensorImplMut::from_slice(&sizes, &mut data, &dim_order, &strides).unwrap();

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
            let tensor_impl = unsafe {
                TensorImpl::from_ptr(&sizes, data.as_ptr(), &dim_order, &strides).unwrap()
            };
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
        let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();
        let tensor = Tensor::new(&tensor_impl);

        assert!(tensor.get(&[4, 0, 0]).is_none());
        assert!(tensor.get(&[0, 5, 0]).is_none());
        assert!(tensor.get(&[0, 0, 3]).is_none());
        assert!(tensor.get_as_typed::<i32>(&[4, 0, 0]).is_none());
        assert!(tensor.get_as_typed::<i32>(&[0, 5, 0]).is_none());
        assert!(tensor.get_as_typed::<i32>(&[0, 0, 3]).is_none());

        for (x, y, z) in indices.clone() {
            let actual1 = tensor[&[x, y, z]];
            let actual2 = tensor.get(&[x, y, z]).unwrap();
            let actual3 = tensor.get_as_typed::<i32>(&[x, y, z]).unwrap();
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(actual1, expected);
            assert_eq!(*actual2, expected);
            assert_eq!(*actual3, expected);
        }

        let tensor = tensor.as_type_erased();
        for (x, y, z) in indices.clone() {
            let actual = tensor.get_as_typed::<i32>(&[x, y, z]).unwrap();
            assert_eq!(*actual, x as i32 * 1337 - y as i32 * 87 + z as i32 * 13);
        }
        assert!(tensor.get_as_typed::<f32>(&[0, 0, 0]).is_none())
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
        let mut tensor_impl =
            TensorImplMut::from_slice(&sizes, &mut data, &dim_order, &strides).unwrap();
        let mut tensor = TensorMut::new(&mut tensor_impl);

        assert!(tensor.get_mut(&[4, 0, 0]).is_none());
        assert!(tensor.get_mut(&[0, 5, 0]).is_none());
        assert!(tensor.get_mut(&[0, 0, 3]).is_none());
        assert!(tensor.get_as_typed_mut::<i32>(&[4, 0, 0]).is_none());
        assert!(tensor.get_as_typed_mut::<i32>(&[0, 5, 0]).is_none());
        assert!(tensor.get_as_typed_mut::<i32>(&[0, 0, 3]).is_none());

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
            *tensor.get_as_typed_mut::<i32>(&[x, y, z]).unwrap() =
                x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
        }
        for (x, y, z) in indices.clone() {
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(tensor[&[x, y, z]], expected);
        }
        for (x, y, z) in indices.clone() {
            *tensor.get_as_typed_mut::<i32>(&[x, y, z]).unwrap() = 0;
        }

        // try_get_mut of type-erased tensor
        let mut tensor = tensor.as_type_erased_mut();
        for (x, y, z) in indices.clone() {
            assert_eq!(*tensor.get_as_typed_mut::<i32>(&[x, y, z]).unwrap(), 0);
        }
        for (x, y, z) in indices.clone() {
            *tensor.get_as_typed_mut::<i32>(&[x, y, z]).unwrap() =
                x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
        }
        for (x, y, z) in indices.clone() {
            let expected = x as i32 * 1337 - y as i32 * 87 + z as i32 * 13;
            assert_eq!(
                *tensor.get_as_typed_mut::<i32>(&[x, y, z]).unwrap(),
                expected
            );
        }
        for (x, y, z) in indices.clone() {
            *tensor.get_as_typed_mut::<i32>(&[x, y, z]).unwrap() = 0;
        }
        assert!(tensor.get_as_typed_mut::<f32>(&[0, 0, 0]).is_none())
    }

    #[cfg(feature = "tensor-ptr")]
    #[test]
    fn into_type_erased() {
        let tensor_ptr = TensorPtr::from_vec(vec![1_i32, 2, 3, 4]);
        let tensor = tensor_ptr.as_tensor();
        let tensor = tensor.into_type_erased();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let _ = tensor.into_typed::<i32>();

        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
        let tensor = tensor_ptr.as_tensor_mut();
        let tensor = tensor.into_type_erased();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let mut tensor = tensor.into_typed::<i32>();
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

        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
        let mut tensor = tensor_ptr.as_tensor_mut();
        let mut tensor = tensor.as_type_erased_mut();
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        let mut tensor = tensor.as_typed_mut::<i32>();
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

        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
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

        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
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

        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
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

        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
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
        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<i32>>::from_vec(vec![1, 2, 3, 4])
            .build_mut()
            .unwrap();
        let mut tensor = tensor_ptr.as_tensor_mut().into_type_erased();
        let _ = tensor.as_typed_mut::<f64>();
    }

    #[test]
    fn invalid_strides_or_dim_order() {
        assert!(TensorImpl::from_slice(&[3], &[0; 3], &[0], &[1]).is_ok());
        assert!(TensorImpl::from_slice(&[3], &[0; 3], &[1], &[1]).is_err());
        assert!(TensorImpl::from_slice(&[3], &[0; 30], &[0], &[10]).is_err());

        assert!(TensorImpl::from_slice(&[2, 3], &[0; 6], &[0, 1], &[3, 1]).is_ok());
        assert!(TensorImpl::from_slice(&[2, 3], &[0; 6], &[1, 0], &[3, 1]).is_err());
        assert!(TensorImpl::from_slice(&[2, 3], &[0; 6], &[1, 0], &[1, 2]).is_ok());
        assert!(TensorImpl::from_slice(&[2, 3], &[0; 6], &[0, 1], &[1, 2]).is_err());
        assert!(TensorImpl::from_slice(&[2, 3], &[0; 12], &[1, 0], &[2, 4]).is_err());

        assert!(TensorImplMut::from_slice(&[3], &mut [0; 3], &[0], &[1]).is_ok());
        assert!(TensorImplMut::from_slice(&[3], &mut [0; 3], &[1], &[1]).is_err());
        assert!(TensorImplMut::from_slice(&[3], &mut [0; 30], &[0], &[10]).is_err());

        assert!(TensorImplMut::from_slice(&[2, 3], &mut [0; 6], &[0, 1], &[3, 1]).is_ok());
        assert!(TensorImplMut::from_slice(&[2, 3], &mut [0; 6], &[1, 0], &[3, 1]).is_err());
        assert!(TensorImplMut::from_slice(&[2, 3], &mut [0; 6], &[1, 0], &[1, 2]).is_ok());
        assert!(TensorImplMut::from_slice(&[2, 3], &mut [0; 6], &[0, 1], &[1, 2]).is_err());
        assert!(TensorImplMut::from_slice(&[2, 3], &mut [0; 12], &[1, 0], &[2, 4]).is_err());
    }

    #[test]
    fn scalar_tensor() {
        let scalar = 42;
        let tensor = TensorImpl::from_scalar(&scalar);
        let storage = storage!(Tensor<i32>);
        let tensor = Tensor::new_in_storage(&tensor, storage);
        assert_eq!(tensor.nbytes(), 4);
        assert_eq!(tensor.dim(), 0);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[]);
        assert_eq!(tensor.dim_order(), &[]);
        assert_eq!(tensor.strides(), &[]);
        assert_eq!(unsafe { *tensor.as_ptr() }, 42);
        assert_eq!(tensor[&[]], 42);

        let mut scalar = 17;
        let mut tensor = TensorImplMut::from_scalar(&mut scalar);
        let storage = storage!(TensorMut<i32>);
        let mut tensor = TensorMut::new_in_storage(&mut tensor, storage);
        assert_eq!(tensor.nbytes(), 4);
        assert_eq!(tensor.dim(), 0);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[]);
        assert_eq!(tensor.dim_order(), &[]);
        assert_eq!(tensor.strides(), &[]);
        assert_eq!(unsafe { *tensor.as_ptr() }, 17);
        assert_eq!(tensor[&[]], 17);
        tensor[&[]] = 6;
        assert_eq!(unsafe { *tensor.as_ptr() }, 6);
        assert_eq!(tensor[&[]], 6);
    }
}
