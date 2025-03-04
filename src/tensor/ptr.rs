use executorch_sys::cxx::{self, vector::VectorElement, ExternType, SharedPtr, UniquePtr};

use super::*;

/// A smart pointer type for managing the lifetime of a Tensor.
///
/// Under the hood this struct is a wrapper around a `cxx::SharedPtr<Tensor>`.
/// In addition to the regular smart pointer functionality that allows to clone and drop the pointer,
/// the Cpp `shared_ptr` is also used to manage the lifetime of allocations a Tensor is usually depends on,
/// such as [`TensorImpl`](super::TensorImpl), the data buffer and the sizes, dim order and strides arrays.
/// This allows a much more user-friendly API for creating and managing Tensors:
/// ```rust,ignore
/// let mut module = Module::new(...);
///
/// // Create a TensorPtr from an ndarray, clean and short syntax
/// let tensor_ptr = TensorPtr::from_array(array![1.0_f32]);
/// let outputs = module.forward(&[tensor_ptr.into_evalue()]).unwrap();
///
/// // Alternatively, manage the lifetimes yourself:
///
/// // Create a Tensor from an ndarray and manage the lifetime of the TensorImpl on the stack
/// let array_storate = ArrayStorage::new(array![1.0_f32]);
/// let tensor_impl = array_storate.as_tensor_impl();
/// let tensor = Tensor::new(&tensor_impl);
/// let outputs = module.forward(&[tensor.into_evalue()]).unwrap();
///
/// // Create a Tensor from raw data/sizes/dim_order/strides arrays and manage all lifetimes on the stack
/// let data = [1.0_f32];
/// let sizes = [1];
/// let dim_order = [0];
/// let strides = [1];
/// let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides);
/// let tensor = Tensor::new(&tensor);
/// let outputs = module.forward(&[tensor.into_evalue()]).unwrap();
/// ```
#[derive(Clone)]
pub struct TensorPtr<'a, D: Data>(SharedPtr<et_c::cpp::Tensor>, PhantomData<(&'a (), D)>);
impl<S: Scalar> TensorPtr<'static, View<S>> {
    /// Create a new [`TensorPtr`] from an [`Array`](ndarray::Array).
    ///
    /// To create a mutable tensor from an array, use [`TensorPtrBuilder`].
    #[cfg(feature = "ndarray")]
    pub fn from_array<D: ndarray::Dimension>(array: ndarray::Array<S, D>) -> Self {
        TensorPtrBuilder::<View<S>>::from_array(array).build()
    }

    /// Create a one dimensional [`TensorPtr`] from a vector.
    ///
    /// To create a mutable tensor from a vector, use [`TensorPtrBuilder`].
    pub fn from_vec(vec: Vec<S>) -> Self {
        TensorPtrBuilder::<View<S>>::from_vec(vec).build()
    }
}
impl<'a, S: Scalar> TensorPtr<'a, View<S>> {
    /// Create a new [`TensorPtr`] from an [`Array`](ndarray::Array).
    ///
    /// To create a mutable tensor from an array view, use [`TensorPtrBuilder`].
    #[cfg(feature = "ndarray")]
    pub fn from_array_view<D: ndarray::Dimension>(array: ndarray::ArrayView<'a, S, D>) -> Self {
        TensorPtrBuilder::<View<S>>::from_array_view(array).build()
    }

    /// Create a one dimensional [`TensorPtr`] from a slice.
    ///
    /// To create a mutable tensor from a slice, use [`TensorPtrBuilder`].
    pub fn from_slice(data: &'a [S]) -> Self {
        TensorPtrBuilder::<View<S>>::from_slice(data).build()
    }
}
impl<D: Data> TensorPtr<'_, D> {
    /// Get an immutable tensor that points to the underlying data.
    pub fn as_tensor(&self) -> TensorBase<D::Immutable> {
        let tensor = self.0.as_ref().unwrap();
        // Safety: *et_c::cpp::Tensor and et_c::Tensor is the same.
        let tensor = unsafe {
            std::mem::transmute::<*const et_c::cpp::Tensor, et_c::TensorRef>(tensor as *const _)
        };
        // Safety: the tensor is valid and the data is immutable.
        unsafe { TensorBase::convert_from(TensorBase::from_inner_ref(tensor)) }
    }

    /// Get a mutable tensor that points to the underlying data.
    pub fn as_tensor_mut(&mut self) -> TensorBase<D>
    where
        D: DataMut,
    {
        let tensor = self.0.as_ref().unwrap();
        let tensor = et_c::TensorRef {
            ptr: tensor as *const et_c::cpp::Tensor as *const _,
        };
        // Safety: the tensor is mutable, and we are the sole borrower.
        unsafe { TensorBase::convert_from(TensorBase::from_inner_ref(tensor)) }
    }
}

/// A builder for creating a [`TensorPtr`].
pub struct TensorPtrBuilder<'a, D: DataTyped> {
    sizes: UniquePtr<cxx::Vector<SizesType>>,
    data: TensorPtrBuilderData<'a, D>,
    strides: Option<UniquePtr<cxx::Vector<StridesType>>>,
    dynamism: et_c::TensorShapeDynamism,
}
enum TensorPtrBuilderData<'a, D: DataTyped> {
    Vec { data: Vec<D::Scalar>, offset: usize },
    Slice(&'a [D::Scalar]),
    SliceMut(&'a mut [D::Scalar]),
    Ptr(*const D::Scalar, PhantomData<&'a ()>),
    PtrMut(*mut D::Scalar, PhantomData<&'a ()>),
}
impl<D: DataTyped> TensorPtrBuilder<'static, D> {
    /// Create a new builer from an [`Array`](ndarray::Array).
    ///
    /// The sizes and strides are extracted from the array, and the data is moved (without a copy) into the tensor
    /// builder.
    ///
    /// This function can be used to create both immutable and mutable tensors, as the builder owns the array data.
    /// Use [`build`](Self::build) or [`build_mut`](Self::build_mut) accordingly.
    /// ```rust,ignore
    /// let immutable_tensor = TensorPtrBuilder::<View<f32>>::from_array(array![1.0]).build();
    /// let mutable_tensor = TensorPtrBuilder::<ViewMut<f32>>::from_array(array![1.0]).build_mut();
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn from_array<Dim: ndarray::Dimension>(array: ndarray::Array<D::Scalar, Dim>) -> Self {
        Self {
            sizes: cxx_vec(array.shape().iter().map(|&s| s as SizesType)),
            strides: Some(cxx_vec(
                ndarray::ArrayBase::strides(&array)
                    .iter()
                    .map(|&s| s as StridesType),
            )),
            data: {
                let (data, data_offset) = array.into_raw_vec_and_offset();
                let data_offset = data_offset.unwrap_or(0);
                assert!(data_offset < data.len());

                TensorPtrBuilderData::Vec {
                    data,
                    offset: data_offset,
                }
            },
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }

    /// Create a one dimensional builer from a vector.
    ///
    /// The dimensions and strides are initialized to `[data.len()]`, `[1]` respectively, but can be changed with the
    /// [`sizes`](Self::sizes) and [`strides`](Self::strides) methods.
    ///
    /// This function can be used to create both immutable and mutable tensors, as the builder owns the vector data.
    /// Use [`build`](Self::build) or [`build_mut`](Self::build_mut) accordingly.
    /// ```rust,ignore
    /// let immutable_tensor = TensorPtrBuilder::<View<f32>>::from_vec(vec![1.0]).build();
    /// let mutable_tensor = TensorPtrBuilder::<ViewMut<f32>>::from_vec(vec![1.0]).build_mut();
    /// ```
    pub fn from_vec(data: Vec<D::Scalar>) -> Self {
        Self {
            sizes: cxx_vec([data.len() as SizesType]),
            data: TensorPtrBuilderData::Vec { data, offset: 0 },
            strides: None,
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }
}
impl<'a, S: Scalar> TensorPtrBuilder<'a, View<S>> {
    /// Create a new builer from an [`ArrayView`](ndarray::ArrayView).
    ///
    /// The sizes and strides are extracted from the array, and a pointer to the data is stored in the tensor builder.
    #[cfg(feature = "ndarray")]
    pub fn from_array_view<Dim: ndarray::Dimension>(array: ndarray::ArrayView<'a, S, Dim>) -> Self {
        Self {
            sizes: cxx_vec(array.shape().iter().map(|&s| s as SizesType)),
            data: TensorPtrBuilderData::Ptr(array.as_ptr(), PhantomData),
            strides: Some(cxx_vec(
                ndarray::ArrayBase::strides(&array)
                    .iter()
                    .map(|&s| s as StridesType),
            )),
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }

    /// Create a builder of a one dimensional tensor from a slice.
    ///
    /// The dimensions and strides are initialized to `[data.len()]`, `[1]` respectively, but can be changed with the
    /// [`sizes`](Self::sizes) and [`strides`](Self::strides) methods.
    pub fn from_slice(data: &'a [S]) -> Self {
        Self {
            sizes: cxx_vec([data.len() as SizesType]),
            data: TensorPtrBuilderData::Slice(data),
            strides: None,
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }

    /// Create a builder from a data pointer.
    ///
    /// Arguments:
    /// - `data`: a pointer to the data.
    /// - `sizes`: the dimensions of the tensor.
    ///
    /// The strides are initialized to `[sizes[-2]*...*sizes[0], sizes[-3]*...*sizes[0], ..., sizes[0], 1]`,
    /// but can be changed with the [`strides`](Self::strides) method.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the data pointer is valid and and remains valid for the lifetime of the tensor, and
    /// that it points to a valid memory location that can be read according to the sizes and strides.
    pub unsafe fn from_ptr(data: *const S, sizes: impl IntoIterator<Item = SizesType>) -> Self {
        Self {
            data: TensorPtrBuilderData::Ptr(data, PhantomData),
            strides: None,
            sizes: cxx_vec(sizes),
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }
}
impl<'a, S: Scalar> TensorPtrBuilder<'a, ViewMut<S>> {
    /// Create a new builer from an [`ArrayViewMut`](ndarray::ArrayViewMut).
    ///
    /// The sizes and strides are extracted from the array, and a mutable pointer to the data is stored in the tensor
    /// builder.
    #[cfg(feature = "ndarray")]
    pub fn from_array_view_mut<Dim: ndarray::Dimension>(
        mut array: ndarray::ArrayViewMut<'a, S, Dim>,
    ) -> Self {
        Self {
            sizes: cxx_vec(array.shape().iter().map(|&s| s as SizesType)),
            data: TensorPtrBuilderData::PtrMut(array.as_mut_ptr(), PhantomData),
            strides: Some(cxx_vec(
                ndarray::ArrayBase::strides(&array)
                    .iter()
                    .map(|&s| s as StridesType),
            )),
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }

    /// Create a builder of a one dimensional tensor from a mutable slice.
    ///
    /// The dimensions and strides are initialized to `[data.len()]`, `[1]` respectively, but can be changed with the
    /// [`sizes`](Self::sizes) and [`strides`](Self::strides) methods.
    pub fn from_slice_mut(data: &'a mut [S]) -> Self {
        Self {
            sizes: cxx_vec([data.len() as SizesType]),
            data: TensorPtrBuilderData::SliceMut(data),
            strides: None,
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }

    /// Create a builder from a mutable data pointer.
    ///
    /// Arguments:
    /// - `data`: a mutable pointer to the data.
    /// - `sizes`: the dimensions of the tensor.
    ///
    /// The strides are initialized to `[sizes[-2]*...*sizes[0], sizes[-3]*...*sizes[0], ..., sizes[0], 1]`,
    /// but can be changed with the [`strides`](Self::strides) method.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the data pointer is valid and and remains valid for the lifetime of the tensor, and
    /// that it points to a valid memory location that can be read and written to according to the sizes and strides.
    pub unsafe fn from_ptr_mut(data: *mut S, sizes: impl IntoIterator<Item = SizesType>) -> Self {
        Self {
            data: TensorPtrBuilderData::PtrMut(data, PhantomData),
            strides: None,
            sizes: cxx_vec(sizes),
            dynamism: et_c::TensorShapeDynamism::TensorShapeDynamism_STATIC,
        }
    }
}
impl<'a, D: DataTyped> TensorPtrBuilder<'a, D> {
    /// Set the dimensions of the tensor.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the sizes are valid and make sense with respect to the data buffer and strides,
    /// namely that that the number of dimensions match the strides and that accessing the data buffer with
    /// any index according to the sizes and strides is valid.
    pub unsafe fn sizes(mut self, sizes: impl IntoIterator<Item = SizesType>) -> Self {
        self.sizes = cxx_vec(sizes);
        self
    }

    /// Set the strides of the tensor.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the strides are valid and make sense with respect to the data buffer and sizes,
    /// namely that that the number of dimensions match the strides and that accessing the data buffer with
    /// any index according to the sizes and strides is valid.
    pub unsafe fn strides(mut self, strides: impl IntoIterator<Item = StridesType>) -> Self {
        self.strides = Some(cxx_vec(strides));
        self
    }

    /// Build an immutable tensor.
    ///
    /// # Panics
    ///
    /// The function panics if the number of dimensions in the sizes and strides array do not match.
    /// The function may panic if the sizes and  strides do not make sense with respect to the data buffer,
    /// but this is not guaranteed.
    #[track_caller]
    pub fn build(self) -> TensorPtr<'a, View<D::Scalar>> {
        let ndim = self.sizes.len();
        let dim_order = cxx_vec((0..ndim).map(|s| s as DimOrderType));
        let strides = self.strides.unwrap_or_else(|| default_strides(&self.sizes));
        assert_eq!(ndim, dim_order.len(), "Invalid dim order length");
        assert_eq!(ndim, strides.len(), "Invalid strides length");

        let (data_ptr, allocation_vec, _data_bound) = match self.data {
            TensorPtrBuilderData::Vec { data, offset } => {
                let bound = data.len().checked_sub(offset).unwrap();
                let ptr = unsafe { data.as_ptr().add(offset) };
                (ptr, data, Some(bound))
            }
            TensorPtrBuilderData::Slice(data) => (data.as_ptr(), Vec::new(), Some(data.len())),
            TensorPtrBuilderData::SliceMut(data) => (data.as_ptr(), Vec::new(), Some(data.len())),
            TensorPtrBuilderData::Ptr(data, _) => (data, Vec::new(), None),
            TensorPtrBuilderData::PtrMut(data, _) => (data as *const _, Vec::new(), None),
        };

        // TODO: check sizes, dim_order and strides make sense with respect to the data_bound

        let tensor = unsafe {
            executorch_sys::cpp::TensorPtr_new(
                self.sizes,
                data_ptr as *const u8 as *mut u8,
                dim_order,
                strides,
                D::Scalar::TYPE.cpp(),
                self.dynamism.clone(),
                Box::new(executorch_sys::cpp::util::RustAny::new(Box::new(
                    allocation_vec,
                ))),
            )
        };
        TensorPtr(tensor, PhantomData)
    }

    /// Build a mutable tensor.
    ///
    /// # Panics
    ///
    /// The function panics if the number of dimensions in the sizes and strides arrays do not match.
    /// The function may panic if the sizes and  strides do not make sense with respect to the data buffer,
    /// but this is not guaranteed.
    #[track_caller]
    pub fn build_mut(self) -> TensorPtr<'a, ViewMut<D::Scalar>>
    where
        D: DataMut,
    {
        let ndim = self.sizes.len();
        let dim_order = cxx_vec((0..ndim).map(|s| s as DimOrderType));
        let strides = self.strides.unwrap_or_else(|| default_strides(&self.sizes));
        assert_eq!(ndim, dim_order.len(), "Invalid dim order length");
        assert_eq!(ndim, strides.len(), "Invalid strides length");

        let (data_ptr, allocation_vec, _data_bound) = match self.data {
            TensorPtrBuilderData::Vec { mut data, offset } => {
                let bound = data.len().checked_sub(offset).unwrap();
                let ptr = unsafe { data.as_mut_ptr().add(offset) };
                (ptr, data, Some(bound))
            }
            TensorPtrBuilderData::Slice(_) => {
                panic!("Cannot create a mutable tensor from an immutable slice")
            }
            TensorPtrBuilderData::SliceMut(data) => {
                (data.as_mut_ptr(), Vec::new(), Some(data.len()))
            }
            TensorPtrBuilderData::Ptr(_, _) => {
                panic!("Cannot create a mutable tensor from an immutable pointer")
            }
            TensorPtrBuilderData::PtrMut(data, _) => (data, Vec::new(), None),
        };

        // TODO: check sizes, dim_order and strides make sense with respect to the data_bound

        let tensor = unsafe {
            executorch_sys::cpp::TensorPtr_new(
                self.sizes,
                data_ptr as *const u8 as *mut u8,
                dim_order,
                strides,
                D::Scalar::TYPE.cpp(),
                self.dynamism.clone(),
                Box::new(executorch_sys::cpp::util::RustAny::new(Box::new(
                    allocation_vec,
                ))),
            )
        };
        TensorPtr(tensor, PhantomData)
    }
}

fn cxx_vec<T>(elms: impl IntoIterator<Item = T>) -> UniquePtr<cxx::Vector<T>>
where
    T: ExternType<Kind = cxx::kind::Trivial> + VectorElement,
{
    let mut vec = cxx::Vector::new();
    elms.into_iter().for_each(|e| vec.pin_mut().push(e));
    vec
}

fn default_strides(sizes: &cxx::Vector<SizesType>) -> UniquePtr<cxx::Vector<StridesType>> {
    let mut strides = cxx_vec::<SizesType>((0..sizes.len()).map(|_| 0));
    let mut stride = 1;
    for i in (0..sizes.len()).rev() {
        strides.as_mut().unwrap().index_mut(i).unwrap().set(stride);
        stride *= sizes.get(i).unwrap();
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ndarray")]
    #[test]
    fn fron_array() {
        let array = ndarray::array![[1, 2], [3, 4]];
        let tensor_ptr = TensorPtr::from_array(array.clone());
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(array, tensor.as_array::<ndarray::Ix2>());
    }

    #[test]
    fn fron_vec() {
        let vec = vec![1, 2, 3, 4];
        let tensor_ptr = TensorPtr::from_vec(vec.clone());
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(
            vec,
            (0..vec.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn fron_array_view() {
        let array = ndarray::array![[1, 2], [3, 4]];
        let tensor_ptr = TensorPtr::from_array_view(array.view());
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(array, tensor.as_array::<ndarray::Ix2>());
    }

    #[test]
    fn fron_slice() {
        let data = [1, 2, 3, 4];
        let tensor_ptr = TensorPtr::from_slice(&data);
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(
            data.to_vec(),
            (0..data.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn as_tensor_mut() {
        let mut data = [1, 2, 3, 4];
        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<_>>::from_slice_mut(&mut data).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut();
        tensor[&[2]] = 50;
        drop(tensor);
        assert_eq!(data, [1, 2, 50, 4]);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn builer_from_array() {
        let array = ndarray::array![[1, 2], [3, 4]];
        let tensor_ptr = TensorPtrBuilder::<View<_>>::from_array(array.clone()).build();
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(array, tensor.as_array::<ndarray::Ix2>());
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn builer_from_array_build_mut() {
        let array = ndarray::array![[1, 2], [3, 4]];
        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<_>>::from_array(array.clone()).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut();
        assert_eq!(array, tensor.as_array::<ndarray::Ix2>());
        tensor[&[1, 1]] = 50;
        assert_ne!(array, tensor.as_array::<ndarray::Ix2>());
        assert_eq!(
            tensor.as_array::<ndarray::Ix2>(),
            ndarray::array![[1, 2], [3, 50]]
        );
    }

    #[test]
    fn builer_from_vec() {
        let vec = vec![1, 2, 3, 4];
        let tensor_ptr = TensorPtrBuilder::<View<_>>::from_vec(vec.clone()).build();
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(
            vec,
            (0..vec.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn builer_from_vec_build_mut() {
        let vec = vec![1, 2, 3, 4];
        let mut tensor_ptr = TensorPtrBuilder::<ViewMut<_>>::from_vec(vec.clone()).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut();
        assert_eq!(
            vec,
            (0..vec.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
        tensor[&[2]] = 50;
        assert_eq!(
            vec![1, 2, 50, 4],
            (0..vec.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn builer_from_array_view() {
        let array = ndarray::array![[1, 2], [3, 4]];
        let tensor_ptr = TensorPtrBuilder::from_array_view(array.view()).build();
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(array, tensor.as_array::<ndarray::Ix2>());
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn builer_from_array_view_mut() {
        let array_orig = ndarray::array![[1, 2], [3, 4]];
        let mut array = array_orig.clone();
        let mut tensor_ptr = TensorPtrBuilder::from_array_view_mut(array.view_mut()).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut();
        assert_eq!(array_orig, tensor.as_array::<ndarray::Ix2>());
        tensor[&[1, 1]] = 50;
        assert_eq!(
            tensor.as_array::<ndarray::Ix2>(),
            ndarray::array![[1, 2], [3, 50]]
        );
        drop(tensor);
        assert_eq!(array, ndarray::array![[1, 2], [3, 50]]);
    }

    #[test]
    fn builer_from_slice() {
        let data = [1, 2, 3, 4];
        let tensor_ptr = TensorPtrBuilder::from_slice(&data).build();
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(
            data.to_vec(),
            (0..data.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn builer_from_slice_mut() {
        let data_orig = [1, 2, 3, 4];
        let mut data = data_orig;
        let mut tensor_ptr = TensorPtrBuilder::from_slice_mut(&mut data).build_mut();
        let mut tensor = tensor_ptr.as_tensor_mut();
        assert_eq!(
            data_orig.to_vec(),
            (0..data_orig.len())
                .map(|i| tensor[&[i]])
                .collect::<Vec<_>>()
        );
        tensor[&[2]] = 50;
        assert_eq!(
            vec![1, 2, 50, 4],
            (0..data_orig.len())
                .map(|i| tensor[&[i]])
                .collect::<Vec<_>>()
        );
        drop(tensor);
        assert_eq!([1, 2, 50, 4], data);
    }

    #[test]
    fn builer_from_ptr() {
        let data = [1, 2, 3, 4];
        let tensor_ptr =
            unsafe { TensorPtrBuilder::from_ptr(data.as_ptr(), [data.len() as SizesType]).build() };
        let tensor = tensor_ptr.as_tensor();
        assert_eq!(
            data.to_vec(),
            (0..data.len()).map(|i| tensor[&[i]]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn builer_from_ptr_mut() {
        let data_orig = [1, 2, 3, 4];
        let mut data = data_orig;
        let mut tensor_ptr = unsafe {
            TensorPtrBuilder::from_ptr_mut(data.as_mut_ptr(), [data.len() as SizesType]).build_mut()
        };
        let mut tensor = tensor_ptr.as_tensor_mut();
        assert_eq!(
            data_orig.to_vec(),
            (0..data_orig.len())
                .map(|i| tensor[&[i]])
                .collect::<Vec<_>>()
        );
        tensor[&[2]] = 50;
        assert_eq!(
            vec![1, 2, 50, 4],
            (0..data_orig.len())
                .map(|i| tensor[&[i]])
                .collect::<Vec<_>>()
        );
        drop(tensor);
        assert_eq!([1, 2, 50, 4], data);
    }
}
