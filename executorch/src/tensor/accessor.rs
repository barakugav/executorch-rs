use std::ops::{Index, IndexMut};

use super::{SizesType, StridesType};

pub(crate) struct TensorAccessorInner<'a, T, const N: usize> {
    data: *const T,
    sizes: [SizesType; N],
    strides: [StridesType; N],
    phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T, const N: usize> TensorAccessorInner<'a, T, N> {
    pub(crate) unsafe fn new(
        data: *const T,
        sizes: &'a [SizesType],
        strides: &'a [StridesType],
    ) -> Self {
        Self {
            data,
            sizes: sizes.try_into().unwrap(),
            strides: strides.try_into().unwrap(),
            phantom: std::marker::PhantomData,
        }
    }

    fn offset_of(&self, index: [usize; N]) -> Option<isize> {
        let valid_index = index
            .iter()
            .zip(self.sizes)
            .all(|(&idx, size)| idx < size as usize);
        valid_index.then(|| unsafe { self.offset_of_unchecked(index) })
    }

    unsafe fn offset_of_unchecked(&self, index: [usize; N]) -> isize {
        let mut offset = 0isize;
        for (&idx, stride) in index.iter().zip(self.strides) {
            offset += idx as isize * stride as isize;
        }
        offset
    }
}

/// A fast accessor for a tensor.
///
/// The accessor is a utility struct, templated over the type of the tensor elements and the number
/// of dimensions, which make it very efficient to access tensor elements by index.
/// A regular Tensor stores its number of dimensions dynamically, and its typed can be stored
/// dynamically or known at compile time
/// (see [Tensor][`crate::tensor::Tensor`] and [TensorAny][`crate::tensor::TensorAny`]).
/// If you know the rank (number of dimensions) and the type of the tensor elements at compile time,
/// you can use this accessor to access the tensor elements efficiently.
pub struct TensorAccessor<'a, T, const N: usize>(pub(crate) TensorAccessorInner<'a, T, N>);
impl<'a, T, const N: usize> TensorAccessor<'a, T, N> {
    /// Get a reference to the tensor element at the given index.
    ///
    /// Returns the element at the given index, or `None` if the index is out of bounds.
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        let offset = self.0.offset_of(index)?;
        Some(unsafe { &*self.0.data.offset(offset) })
    }

    /// Get a reference to the tensor element at the given index, without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = unsafe { self.0.offset_of_unchecked(index) };
        unsafe { &*self.0.data.offset(offset) }
    }
}
impl<'a, T> Index<usize> for TensorAccessor<'a, T, 1> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        self.get([index]).unwrap()
    }
}
impl<'a, T, const N: usize> Index<[usize; N]> for TensorAccessor<'a, T, N> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &Self::Output {
        self.get(index).unwrap()
    }
}

/// A mutable accessor for a tensor.
///
/// The accessor is a utility struct, templated over the type of the tensor elements and the number
/// of dimensions, which make it very efficient to access tensor elements by index.
/// This is similar to [TensorAccessor], but allows for mutable access to the tensor elements.
/// See the immutable accessor for more details.
pub struct TensorAccessorMut<'a, T, const N: usize>(pub(crate) TensorAccessorInner<'a, T, N>);
impl<'a, T, const N: usize> TensorAccessorMut<'a, T, N> {
    /// Get a reference to the tensor element at the given index.
    ///
    /// Returns the element at the given index, or `None` if the index is out of bounds.
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        let offset = self.0.offset_of(index)?;
        Some(unsafe { &*self.0.data.offset(offset) })
    }

    /// Get a reference to the tensor element at the given index, without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = unsafe { self.0.offset_of_unchecked(index) };
        unsafe { &*self.0.data.offset(offset) }
    }

    /// Get a mutable reference to the tensor element at the given index.
    ///
    /// Returns the element at the given index, or `None` if the index is out of bounds.
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        let offset = self.0.offset_of(index)?;
        Some(unsafe { &mut *self.0.data.cast_mut().offset(offset) })
    }

    /// Get a mutable reference to the tensor element at the given index, without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_mut_unchecked(&mut self, index: [usize; N]) -> &mut T {
        let offset = unsafe { self.0.offset_of_unchecked(index) };
        unsafe { &mut *self.0.data.cast_mut().offset(offset) }
    }
}
impl<'a, T> Index<usize> for TensorAccessorMut<'a, T, 1> {
    type Output = T;

    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output {
        self.get([index]).unwrap()
    }
}
impl<'a, T, const N: usize> Index<[usize; N]> for TensorAccessorMut<'a, T, N> {
    type Output = T;

    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        self.get(index).unwrap()
    }
}
impl<'a, T> IndexMut<usize> for TensorAccessorMut<'a, T, 1> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut([index]).unwrap()
    }
}
impl<'a, T, const N: usize> IndexMut<[usize; N]> for TensorAccessorMut<'a, T, N> {
    #[track_caller]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use crate::tensor::{RawTensor, RawTensorImpl};

    #[cfg(feature = "alloc")]
    #[test]
    fn get() {
        let sizes = [2, 3];
        let data: [i32; 6] = [1, 2, 3, 4, 5, 6];
        let dim_order = [0, 1];
        let strides = [3, 1];
        let tensor_impl = unsafe {
            RawTensorImpl::from_ptr(&sizes, data.as_ptr().cast_mut(), &dim_order, &strides).unwrap()
        };
        let tensor = unsafe { RawTensor::new(&tensor_impl) };

        assert!(tensor.accessor::<i8, 2>().is_none());
        assert!(tensor.accessor::<i16, 2>().is_none());
        assert!(tensor.accessor::<i64, 2>().is_none());
        assert!(tensor.accessor::<f32, 2>().is_none());
        assert!(tensor.accessor::<f64, 2>().is_none());
        assert!(tensor.accessor::<i32, 1>().is_none());
        assert!(tensor.accessor::<i32, 3>().is_none());
        assert!(tensor.accessor::<i32, 4>().is_none());
        let accessor = tensor.accessor::<i32, 2>().unwrap();
        assert_eq!(accessor.get([0, 0]), Some(&1));
        assert_eq!(accessor.get([0, 1]), Some(&2));
        assert_eq!(accessor.get([0, 2]), Some(&3));
        assert_eq!(accessor.get([0, 3]), None);
        assert_eq!(accessor.get([1, 0]), Some(&4));
        assert_eq!(accessor.get([1, 1]), Some(&5));
        assert_eq!(accessor.get([1, 2]), Some(&6));
        assert_eq!(accessor.get([2, 0]), None);
        assert_eq!(accessor[[0, 0]], 1);
        assert_eq!(accessor[[0, 1]], 2);
        assert_eq!(accessor[[0, 2]], 3);
        assert_eq!(accessor[[1, 0]], 4);
        assert_eq!(accessor[[1, 1]], 5);
        assert_eq!(accessor[[1, 2]], 6);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn get_mut() {
        let sizes = [2, 3];
        let mut data: [i32; 6] = [1, 2, 3, 4, 5, 6];
        let dim_order = [0, 1];
        let strides = [3, 1];
        let mut tensor_impl = unsafe {
            RawTensorImpl::from_ptr(&sizes, data.as_mut_ptr(), &dim_order, &strides).unwrap()
        };
        #[allow(clippy::unnecessary_mut_passed)]
        let mut tensor = unsafe { RawTensor::new(&mut tensor_impl) };

        unsafe {
            assert!(tensor.accessor_mut::<i8, 2>().is_none());
            assert!(tensor.accessor_mut::<i16, 2>().is_none());
            assert!(tensor.accessor_mut::<i64, 2>().is_none());
            assert!(tensor.accessor_mut::<f32, 2>().is_none());
            assert!(tensor.accessor_mut::<f64, 2>().is_none());
            assert!(tensor.accessor_mut::<i32, 1>().is_none());
            assert!(tensor.accessor_mut::<i32, 3>().is_none());
            assert!(tensor.accessor_mut::<i32, 4>().is_none());
            let mut accessor = tensor.accessor_mut::<i32, 2>().unwrap();
            assert_eq!(accessor.get([0, 0]), Some(&1));
            assert_eq!(accessor.get([0, 1]), Some(&2));
            assert_eq!(accessor.get([0, 2]), Some(&3));
            assert_eq!(accessor.get([0, 3]), None);
            assert_eq!(accessor.get([1, 0]), Some(&4));
            assert_eq!(accessor.get([1, 1]), Some(&5));
            assert_eq!(accessor.get([1, 2]), Some(&6));
            assert_eq!(accessor.get([2, 0]), None);

            *accessor.get_mut([0, 0]).unwrap() = 10;
            accessor[[1, 1]] = 25;
            assert_eq!(accessor[[0, 0]], 10);
            assert_eq!(accessor[[0, 1]], 2);
            assert_eq!(accessor[[0, 2]], 3);
            assert_eq!(accessor[[1, 0]], 4);
            assert_eq!(accessor[[1, 1]], 25);
            assert_eq!(accessor[[1, 2]], 6);
        }
    }
}
