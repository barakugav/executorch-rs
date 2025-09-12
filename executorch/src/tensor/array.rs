use ndarray::{ArrayBase, ArrayView, ArrayViewMut, ShapeBuilder};

use executorch_sys as et_c;

use crate::util::IntoRust;
use crate::{CError, Error, Result};

use super::{
    DataMut, DataTyped, DimOrderType, Scalar, SizesType, StridesType, TensorBase, TensorImpl,
    TensorImplMut,
};

impl<D: DataTyped> TensorBase<'_, D> {
    /// Get an array view of the tensor.
    ///
    /// # Panics
    ///
    /// If the number of dimensions of the tensor does not match the number of dimensions of the type `Dim`
    pub fn as_array<Dim: Dimension>(&self) -> ArrayView<'_, D::Scalar, Dim> {
        if let Some(arr_ndim) = Dim::NDIM {
            let tensor_ndim = self.dim();
            assert_eq!(
                tensor_ndim, arr_ndim,
                "Dimension mismatch: {tensor_ndim} != {arr_ndim}",
            );
        }
        let ndim = self.dim();
        let mut dim = Dim::zeros(ndim);
        let mut strides = Dim::zeros(ndim);
        let mut dim_order = Dim::zeros(ndim);
        for (i, d) in self.sizes().iter().enumerate() {
            dim[i] = *d as usize;
        }
        for (i, s) in self.strides().iter().enumerate() {
            strides[i] = *s as usize;
        }
        for (i, s) in self.dim_order().iter().enumerate() {
            dim_order[i] = *s as usize;
        }
        let ptr = self.as_ptr();
        unsafe { ArrayView::from_shape_ptr(dim.strides(strides), ptr) }.permuted_axes(dim_order)
    }

    /// Get an array view of the tensor with dynamic number of dimensions.
    #[cfg(feature = "alloc")]
    pub fn as_array_dyn(&self) -> ArrayView<'_, D::Scalar, ndarray::IxDyn> {
        self.as_array()
    }
}

impl<'a, D: DataTyped + DataMut> TensorBase<'a, D> {
    /// Get a mutable array view of the tensor.
    ///
    /// # Panics
    ///
    /// If the number of dimensions of the tensor does not match the number of dimensions of the type `Dim`.
    pub fn as_array_mut<Dim: Dimension>(&mut self) -> ArrayViewMut<'a, D::Scalar, Dim> {
        let ndim = self.dim();
        let mut dim = Dim::zeros(ndim);
        let mut strides = Dim::zeros(ndim);
        let mut dim_order = Dim::zeros(ndim);
        for (i, d) in self.sizes().iter().enumerate() {
            dim[i] = *d as usize;
        }
        for (i, s) in TensorBase::strides(self).iter().enumerate() {
            strides[i] = *s as usize;
        }
        for (i, s) in self.dim_order().iter().enumerate() {
            dim_order[i] = *s as usize;
        }
        let ptr = self.as_mut_ptr();
        unsafe { ArrayViewMut::from_shape_ptr(dim.strides(strides), ptr) }.permuted_axes(dim_order)
    }

    /// Get a mutable array view of the tensor with dynamic number of dimensions.
    #[cfg(feature = "alloc")]
    pub fn as_array_mut_dyn(&mut self) -> ArrayViewMut<'a, D::Scalar, ndarray::IxDyn> {
        self.as_array_mut()
    }
}

/// A wrapper around [`ndarray::ArrayBase`] that can be converted to [`TensorImplBase`](super::TensorImplBase).
///
/// The [`TensorImplBase`](super::TensorImplBase) struct does not own any of the data it points to alongside the dimensions and strides arrays.
/// This struct allocate any required auxiliary memory in addition to the underlying `ndarray::ArrayBase`,
/// allowing to create a [`TensorImplBase`](super::TensorImplBase) that points to it.
/// If the number of dimensions is known at compile time, this struct will not allocate any memory on the heap.
///
/// Use [`as_tensor_impl`](ArrayStorage::as_tensor_impl) and [`as_tensor_impl_mut`](ArrayStorage::as_tensor_impl_mut)
/// to obtain a [`TensorImplBase`](super::TensorImplBase) pointing to this array data.
pub struct ArrayStorage<A: Scalar, S: ndarray::RawData<Elem = A>, D: Dimension> {
    array: ArrayBase<S, D>,
    sizes: D::Arr<SizesType>,
    dim_order: D::Arr<DimOrderType>,
    strides: D::Arr<StridesType>,
}
impl<A: Scalar, S: ndarray::RawData<Elem = A>, D: Dimension> ArrayStorage<A, S, D> {
    /// Create a new [`ArrayStorage`] from an ndarray.
    ///
    /// # Errors
    ///
    /// Returns an error if the array is not dense, i.e. if the strides are not the standard layout strides of some
    /// permutation of the dimensions.
    pub fn new(array: ArrayBase<S, D>) -> Result<Self> {
        let ndim = array.ndim();
        let mut sizes = D::Arr::zeros(ndim);
        let mut dim_order = D::Arr::zeros(ndim);
        let mut strides = D::Arr::zeros(ndim);
        for (i, d) in array.shape().iter().enumerate() {
            sizes.as_mut()[i] = *d as SizesType;
        }
        for (i, s) in ndarray::ArrayBase::strides(&array).iter().enumerate() {
            strides.as_mut()[i] = *s as StridesType;
        }

        unsafe {
            et_c::executorch_stride_to_dim_order(
                strides.as_ref().as_ptr(),
                ndim,
                dim_order.as_mut().as_mut_ptr(),
            )
        }
        .rs()?;
        let valid_strides = unsafe {
            et_c::executorch_is_valid_dim_order_and_strides(
                ndim,
                sizes.as_ref().as_ptr(),
                dim_order.as_ref().as_ptr(),
                strides.as_ref().as_ptr(),
            )
        };
        if !valid_strides {
            crate::log::error!("Invalid strides");
            return Err(Error::CError(CError::InvalidArgument));
        }

        Ok(Self {
            array,
            sizes,
            dim_order,
            strides,
        })
    }

    /// Create a [`TensorImpl`] pointing to this struct's data.
    ///
    /// The [`TensorImpl`] does not own the data or the sizes, dim order and strides of the tensor. This struct
    /// must outlive the [`TensorImpl`] created from it.
    pub fn as_tensor_impl(&self) -> TensorImpl<'_, A> {
        unsafe {
            TensorImpl::from_ptr(
                self.sizes.as_ref(),
                self.array.as_ptr(),
                self.dim_order.as_ref(),
                self.strides.as_ref(),
            )
        }
        .unwrap()
    }

    /// Get a reference to the underlying ndarray.
    pub fn as_array(&self) -> &ArrayBase<S, D> {
        &self.array
    }

    /// Extract the inner array out of this wrapper
    pub fn into_array(self) -> ArrayBase<S, D> {
        self.array
    }
}
impl<A: Scalar, S: ndarray::RawDataMut<Elem = A>, D: Dimension> ArrayStorage<A, S, D> {
    /// Create a [`TensorImplMut`] pointing to this struct's data.
    ///
    /// The [`TensorImplMut`] does not own the data or the sizes, dim order and strides of the tensor. This struct
    /// must outlive the [`TensorImplMut`] created from it.
    pub fn as_tensor_impl_mut(&mut self) -> TensorImplMut<'_, A> {
        let tensor = self.as_tensor_impl();
        // Safety: TensorImpl has the same memory layout as TensorImplBase
        unsafe { std::mem::transmute::<TensorImpl<A>, TensorImplMut<A>>(tensor) }
    }
}
impl<A: Scalar, S: ndarray::RawData<Elem = A>, D: Dimension> AsRef<ArrayBase<S, D>>
    for ArrayStorage<A, S, D>
{
    fn as_ref(&self) -> &ArrayBase<S, D> {
        self.as_array()
    }
}

/// An extension to `ndarray::Dimension` for dimensions used to convert to/from Tensors.
pub trait Dimension: ndarray::Dimension {
    /// The array type that holds the sizes, dim order and strides of the tensor.
    ///
    /// Can be either a fixed size array (supported without alloc) or a dynamic array (vector).
    type Arr<T: Clone + Copy + Default>: DimArr<T>;
}
impl<D: FixedSizeDim> Dimension for D {
    type Arr<T: Clone + Copy + Default> = D::Arr<T>;
}
#[cfg(feature = "alloc")]
impl Dimension for ndarray::IxDyn {
    type Arr<T: Clone + Copy + Default> = crate::alloc::Vec<T>;
}

/// An abstraction over fixed-size arrays and regular vectors if the `alloc` feature is enabled.
pub trait DimArr<T>: AsRef<[T]> + AsMut<[T]> {
    /// Create an array of zeros with the given number of dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the given number of dimensions is not supported by the array. For example, if the array is a fixed
    /// size array of size 3, it will panic if the given number of dimensions is 4. Regular vectors will never panic.
    fn zeros(ndim: usize) -> Self;
}

macro_rules! impl_dim_arr {
    (0) => {
        impl<T: Clone + Copy + Default> DimArr<T> for [T; 0] {
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, 0);
                []
            }
        }
    };
    ($size:literal) => {
        impl<T: Clone + Copy + Default> DimArr<T> for [T; $size] {
            fn zeros(ndim: usize) -> Self {
                assert_eq!(ndim, $size);
                [T::default(); $size]
            }
        }
    };
}
impl_dim_arr!(0);
impl_dim_arr!(1);
impl_dim_arr!(2);
impl_dim_arr!(3);
impl_dim_arr!(4);
impl_dim_arr!(5);
impl_dim_arr!(6);

#[cfg(feature = "alloc")]
impl<T: Clone + Copy + Default> DimArr<T> for crate::alloc::Vec<T> {
    fn zeros(ndim: usize) -> Self {
        crate::alloc::Vec::from_iter(std::iter::repeat(T::default()).take(ndim))
    }
}

/// A marker trait for dimensions that have a fixed size.
///
/// This trait is useful for functions that avoid allocations and want to define additional arrays with the same size as
/// a given dimension.
#[cfg(feature = "ndarray")]
pub trait FixedSizeDim: ndarray::Dimension {
    /// An array with the same fixed size as the dimension.
    type Arr<T: Clone + Copy + Default>: DimArr<T>;
    private_decl! {}
}
#[cfg(feature = "ndarray")]
mod fixed_dim_impl {
    use super::*;

    macro_rules! impl_fixed_size_dim {
        ($size:expr) => {
            impl FixedSizeDim for ndarray::Dim<[ndarray::Ix; $size]> {
                type Arr<T: Clone + Copy + Default> = [T; $size];
                private_impl! {}
            }
        };
    }
    impl_fixed_size_dim!(0);
    impl_fixed_size_dim!(1);
    impl_fixed_size_dim!(2);
    impl_fixed_size_dim!(3);
    impl_fixed_size_dim!(4);
    impl_fixed_size_dim!(5);
    impl_fixed_size_dim!(6);
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use ndarray::{arr1, arr2, Array3, Ix3};

    #[allow(unused_imports)]
    use crate::tensor::*;

    #[cfg(feature = "std")]
    #[test]
    fn array_as_tensor() {
        // Create a 1D array and convert it to a tensor
        let array = ArrayStorage::<i32, _, _>::new(arr1(&[1, 2, 3])).unwrap();
        let tensor_impl = array.as_tensor_impl();
        let tensor = Tensor::new(&tensor_impl);
        assert_eq!(tensor.nbytes(), 12);
        assert_eq!(tensor.size(0), 3);
        assert_eq!(tensor.dim(), 1);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[3]);
        assert_eq!(tensor.dim_order(), &[0]);
        assert_eq!(tensor.strides(), &[1]);
        assert_eq!(tensor.as_ptr(), array.as_ref().as_ptr());
        drop(tensor);

        let array = array.into_array();
        assert_eq!(array, arr1(&[1, 2, 3]));

        // Create a 2D array and convert it to a tensor
        let array =
            ArrayStorage::<f64, _, _>::new(arr2(&[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]])).unwrap();
        let tensor_impl = array.as_tensor_impl();
        let tensor = Tensor::new(&tensor_impl);
        assert_eq!(tensor.nbytes(), 48);
        assert_eq!(tensor.size(0), 2);
        assert_eq!(tensor.size(1), 3);
        assert_eq!(tensor.dim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.scalar_type(), ScalarType::Double);
        assert_eq!(tensor.element_size(), 8);
        assert_eq!(tensor.sizes(), &[2, 3]);
        assert_eq!(tensor.dim_order(), &[0, 1]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.as_ptr(), array.as_ref().as_ptr());
        drop(tensor);

        let array = array.into_array();
        assert_eq!(array, arr2(&[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn array_as_tensor_mut() {
        // Create a 1D array and convert it to a tensor
        let mut array = ArrayStorage::<i32, _, _>::new(arr1(&[1, 2, 3])).unwrap();
        let arr_ptr = array.as_ref().as_ptr();
        let mut tensor_impl = array.as_tensor_impl_mut();
        let tensor = TensorMut::new(&mut tensor_impl);
        assert_eq!(tensor.nbytes(), 12);
        assert_eq!(tensor.size(0), 3);
        assert_eq!(tensor.dim(), 1);
        assert_eq!(tensor.numel(), 3);
        assert_eq!(tensor.scalar_type(), ScalarType::Int);
        assert_eq!(tensor.element_size(), 4);
        assert_eq!(tensor.sizes(), &[3]);
        assert_eq!(tensor.dim_order(), &[0]);
        assert_eq!(tensor.strides(), &[1]);
        assert_eq!(tensor.as_ptr(), arr_ptr);

        // Create a 2D array and convert it to a tensor
        let mut array =
            ArrayStorage::<f64, _, _>::new(arr2(&[[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]])).unwrap();
        let arr_ptr = array.as_ref().as_ptr();
        let mut tensor_impl = array.as_tensor_impl_mut();
        let tensor = TensorMut::new(&mut tensor_impl);
        assert_eq!(tensor.nbytes(), 48);
        assert_eq!(tensor.size(0), 2);
        assert_eq!(tensor.size(1), 3);
        assert_eq!(tensor.dim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.scalar_type(), ScalarType::Double);
        assert_eq!(tensor.element_size(), 8);
        assert_eq!(tensor.sizes(), &[2, 3]);
        assert_eq!(tensor.dim_order(), &[0, 1]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.as_ptr(), arr_ptr);
    }

    #[cfg(feature = "std")]
    #[test]
    fn tensor_as_array() {
        let arr1 = ArrayStorage::new(Array3::<f32>::zeros((3, 6, 4))).unwrap();
        let tensor_impl = arr1.as_tensor_impl();
        let tensor = Tensor::new(&tensor_impl);
        let arr2 = tensor.as_array::<Ix3>();
        assert_eq!(arr1.as_ref(), arr2);
        assert_eq!(arr1.as_ref().strides(), arr2.strides());

        cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
            let arr1 = ArrayStorage::new(arr1.as_ref().view().into_dyn()).unwrap();
            let tensor_impl = arr1.as_tensor_impl();
            let tensor = Tensor::new(&tensor_impl);
            let arr2 = tensor.as_array_dyn().into_shape_with_order(vec![18, 4]).unwrap();
            assert_eq!(arr1.as_ref().view().into_shape_with_order(vec![18, 4]).unwrap(), arr2);
            assert_eq!(arr2.strides(), [4, 1]);
        } }
    }

    #[cfg(feature = "std")]
    #[test]
    fn tensor_as_array_mut() {
        let mut arr1 = ArrayStorage::new(Array3::<f32>::zeros((3, 6, 4))).unwrap();
        let arr1_clone = arr1.as_ref().clone();
        let mut tensor_impl = arr1.as_tensor_impl_mut();
        let mut tensor = TensorMut::new(&mut tensor_impl);
        let arr2 = tensor.as_array_mut::<Ix3>();
        assert_eq!(arr1_clone, arr2);
        assert_eq!(arr1_clone.strides(), arr2.strides());

        cfg_if::cfg_if! { if #[cfg(feature = "alloc")] {
            let mut arr1 = arr1_clone.into_dyn();
            let arr1_clone = arr1.clone();
            let mut arr1 = ArrayStorage::new(arr1.view_mut().into_shape_with_order((18, 4)).unwrap()).unwrap();
            let mut tensor_impl = arr1.as_tensor_impl_mut();
            let mut tensor = TensorMut::new(&mut tensor_impl);
            let arr2 = tensor.as_array_mut_dyn();
            assert_eq!(arr1_clone.view().into_shape_with_order(vec![18, 4]).unwrap(), arr2);
            assert_eq!(arr2.strides(), [4, 1]);
        } }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn invalid_strides() {
        use ndarray::{Array, ShapeBuilder};

        assert!(ArrayStorage::new(
            Array::from_shape_vec((3,).strides((1,)), (0..3).collect()).unwrap()
        )
        .is_ok());
        assert!(ArrayStorage::new(
            Array::from_shape_vec((3,).strides((10,)), (0..30).collect()).unwrap()
        )
        .is_err());

        assert!(ArrayStorage::new(
            Array::from_shape_vec((2, 3).strides((3, 1)), (0..6).collect()).unwrap()
        )
        .is_ok());
        assert!(ArrayStorage::new(
            Array::from_shape_vec((2, 3).strides((1, 2)), (0..6).collect()).unwrap()
        )
        .is_ok());
        assert!(ArrayStorage::new(
            Array::from_shape_vec((2, 3).strides((2, 4)), (0..12).collect()).unwrap()
        )
        .is_err());
    }
}
