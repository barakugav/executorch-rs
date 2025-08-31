use core::pin::Pin;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::memory::{Storable, Storage};
use crate::tensor::TensorAny;
use crate::util::IntoCpp;
use crate::{CError, Error, Result};

use executorch_sys as et_c;

use super::{EValue, Tag};

/// Helper class used to correlate EValues in the executor table, with the
/// unwrapped list of the proper type.
///
/// Because values in the runtime's values
/// table can change during execution, we cannot statically allocate list of
/// objects at deserialization. Imagine the serialized list says index 0 in the
/// value table is element 2 in the list, but during execution the value in
/// element 2 changes (in the case of tensor this means the TensorImpl* stored in
/// the tensor changes). To solve this instead they must be created dynamically
/// whenever they are used.
///
/// Practically this struct is not so easy to work with, but thats the one provided
/// by the Cpp library :).
/// The struct consist of two lists:
/// - `wrapped_vals`: a list of `[*const EValue]`, which contain the actual values of
///   list, boxed in `EValue`.
/// - `unwrapped_vals`: a list of `[T]`, initially uninitialized but when a reference
///   to the actual `T` values is required the values are "unwrapped" from the boxed
///   `EValue` into this array, and returned to the user as a slice.
///
/// This struct is used to represent lists of `i64`, `Tensor` and `Option<Tensor>`
/// within an `EValue`, and it is used to initialize such EValues, but rarely should
/// be used for anything else.
/// EValues internally hold such boxed lists, and expose `&[T]` by unwrapping the
/// wrapped values into the `unwrapped_vals`, keeping the reading interface simple.
///
/// ```rust,ignore
/// let (evalue1, evalue2, evalue3) = (EValue::new(42), EValue::new(17), EValue::new(6));
/// let wrapped_vals = EValuePtrList::new([&evalue1, &evalue2, &evalue3]);
/// let unwrapped_vals = executorch::storage!(i64, [3]);
/// let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();
///
/// let evalue = EValue::new(list);
/// assert_eq!(evalue.tag(), Tag::ListInt);
/// assert_eq!(evalue.as_i64_list(), &[42, 17, 6]);
/// ```
pub struct BoxedEvalueList<'a, T: BoxedEvalueListElement<'a>>(
    pub(crate) T::__ListImpl,
    PhantomData<&'a ()>,
);

impl<'a, T: BoxedEvalueListElement<'a>> BoxedEvalueList<'a, T> {
    /// Create a new boxed list of the given type.
    ///
    /// # Arguments
    /// - `wrapped_vals`: a list of `EValue` that contain the actual values of the list. The inner values
    ///   within the `EValue` must match the type `T`.
    /// - `unwrapped_vals`: an allocation of the unwrapped values. The length of the allocation must
    ///   match the length of the `wrapped_vals`.
    ///
    /// # Returns
    /// A new boxed list with the given values, or an error:
    /// - `InvalidArgument`: if the length of the `wrapped_vals` and `unwrapped_vals` do not match.
    /// - `InvalidType`: if the inner values of the `wrapped_vals` do not match the type `T`.
    pub fn new(
        wrapped_vals: &'a EValuePtrList<'_>,
        unwrapped_vals: Pin<&'a mut [Storage<T>]>,
    ) -> Result<Self> {
        let wrapped_vals_slice = wrapped_vals.as_slice();
        if wrapped_vals_slice.len() != unwrapped_vals.len() {
            return Err(Error::CError(CError::InvalidArgument));
        }
        for i in 0..wrapped_vals_slice.len() {
            let elm = wrapped_vals.get(i).unwrap();
            if let Some(elm) = elm {
                if elm.tag() != T::__ELEMENT_TAG {
                    return Err(Error::CError(CError::InvalidType));
                }
            } else if !T::__ALLOW_NULL_ELEMENT {
                return Err(Error::CError(CError::InvalidType));
            }
        }

        let wrapped_vals = et_c::ArrayRefEValuePtr {
            data: wrapped_vals_slice.as_ptr(),
            len: wrapped_vals_slice.len(),
        };

        let list = unsafe { T::__ListImpl::__new(wrapped_vals, unwrapped_vals)? };
        Ok(Self(list, PhantomData))
    }
}

/// A marker trait for types that can be stored in a [`BoxedEvalueList`].
pub trait BoxedEvalueListElement<'a>: Storable {
    /// The tag of inner values within the list.
    #[doc(hidden)]
    const __ELEMENT_TAG: Tag;

    /// Whether the inner values can be `None`.
    #[doc(hidden)]
    const __ALLOW_NULL_ELEMENT: bool;

    /// The Cpp object that represents the list.
    #[doc(hidden)]
    type __ListImpl: __BoxedEvalueListImpl<Element<'a> = Self>;

    private_decl! {}
}

/// A Cpp list object of a specific element type.
#[doc(hidden)]
pub trait __BoxedEvalueListImpl {
    type Element<'a>: BoxedEvalueListElement<'a, __ListImpl = Self>;

    /// Create a new list from the given wrapped and unwrapped values.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the lengths of the wrapped and unwrapped values match,
    /// that the wrapped values are of the correct type, and that both wrapped and unwrapped
    /// arrays are valid for the lifetime of the returned object.
    unsafe fn __new(
        wrapped_vals: et_c::ArrayRefEValuePtr,
        unwrapped_vals: Pin<&mut [Storage<Self::Element<'_>>]>,
    ) -> Result<Self>
    where
        Self: Sized;

    private_decl! {}
}

macro_rules! ptr2ref {
    ($ptr:expr) => {
        $ptr
    };
    ($ptr:expr, $ref_type:path) => {{
        $ref_type {
            ptr: $ptr as *mut _,
        }
    }};
}
macro_rules! impl_boxed_evalue_list {
    ($element:path, $list_impl:path, $unwrapped_span_type:path, $element_tag:ident, $allow_null_element:expr $(, $unwrapped_type:ty)?) => {
        impl<'a> BoxedEvalueListElement<'a> for $element {
            const __ELEMENT_TAG: Tag = Tag::$element_tag;
            const __ALLOW_NULL_ELEMENT: bool = $allow_null_element;
            type __ListImpl = $list_impl;
            private_impl! {}
        }
        impl __BoxedEvalueListImpl for $list_impl {
            type Element<'a> = $element;

            unsafe fn __new(
                wrapped_vals: et_c::ArrayRefEValuePtr,
                unwrapped_vals: Pin<&mut [Storage<Self::Element<'_>>]>,
            ) -> Result<Self> {
                // Safety: we dont move out of the pinned slice.
                let unwrapped_vals = unsafe { unwrapped_vals.get_unchecked_mut() };
                let unwrapped_vals_ptr = unwrapped_vals.as_mut_ptr() as *mut <Self::Element<'_> as Storable>::__Storage;
                Ok(Self {
                    wrapped_vals,
                    unwrapped_vals: {
                        $unwrapped_span_type {
                            data: ptr2ref!(unwrapped_vals_ptr $(, $unwrapped_type)?),
                            len: unwrapped_vals.len(),
                        }
                    },
                })
            }

            private_impl! {}
        }
    };
}
impl_boxed_evalue_list!(i64, et_c::BoxedEvalueListI64, et_c::SpanI64, Int, false);
impl_boxed_evalue_list!(
    Option<TensorAny<'a>>,
    et_c::BoxedEvalueListOptionalTensor,
    et_c::SpanOptionalTensor,
    Tensor,
    true,
    et_c::OptionalTensorRefMut
);
impl_boxed_evalue_list!(
    TensorAny<'a>,
    et_c::BoxedEvalueListTensor,
    et_c::SpanTensor,
    Tensor,
    false,
    et_c::TensorRefMut
);

/// A list of pointers to `EValue`.
///
/// Usually such list is used as an input to a [`BoxedEvalueList`].
pub struct EValuePtrList<'a>(EValuePtrListInner<'a>);
enum EValuePtrListInner<'a> {
    #[cfg(feature = "alloc")]
    Vec(
        (
            crate::alloc::Vec<et_c::EValueRef>,
            // A lifetime for the `*const EValue` values
            PhantomData<&'a ()>,
        ),
    ),
    Slice(
        (
            &'a [et_c::EValueRef],
            // A lifetime for the `*const EValue` values
            PhantomData<&'a ()>,
        ),
    ),
}
impl<'a> EValuePtrList<'a> {
    #[cfg(feature = "alloc")]
    fn new_impl(values: impl IntoIterator<Item = Option<&'a EValue<'a>>>) -> Self {
        let values: crate::alloc::Vec<et_c::EValueRef> = values
            .into_iter()
            .map(|value| {
                value.map(|value| value.cpp()).unwrap_or(et_c::EValueRef {
                    ptr: std::ptr::null(),
                })
            })
            .collect();
        Self(EValuePtrListInner::Vec((values, PhantomData)))
    }

    /// Create a new list with the give values.
    ///
    /// Usually such list is used as an input to a [`BoxedEvalueList`]. In that case the
    /// values should be of the same type.
    ///
    /// This function require a small allocation on the heap. To avoid this allocation, use
    /// [`new_in_storage`][Self::new_in_storage] instead.
    #[cfg(feature = "alloc")]
    pub fn new(values: impl IntoIterator<Item = &'a EValue<'a>>) -> Self {
        Self::new_impl(values.into_iter().map(Some))
    }

    /// Create a new list with the give values, where some values can be `None`.
    ///
    /// Usually such list is used as an input to a [`BoxedEvalueList`]. In that case the
    /// values should be of the same type, and only `Option<TensorAny>` support `None` values.
    ///
    /// This function require a small allocation on the heap. To avoid this allocation, use
    /// [`new_optional_in_storage`][Self::new_optional_in_storage] instead.
    #[cfg(feature = "alloc")]
    pub fn new_optional(values: impl IntoIterator<Item = Option<&'a EValue<'a>>>) -> Self {
        Self::new_impl(values)
    }

    fn new_in_storage_impl(
        values: impl IntoIterator<Item = Option<&'a EValue<'a>>>,
        storage: Pin<&'a mut [Storage<EValuePtrListElem>]>,
    ) -> Self {
        let mut values = values.into_iter();
        // Safety: we dont move out of the pinned slice.
        let storage = unsafe { storage.get_unchecked_mut() };
        // Safety: Storage<T> is transparent MaybeUninit<<T as Storable>::__Storage>
        let storage = unsafe {
            std::mem::transmute::<
                &mut [Storage<EValuePtrListElem>],
                &mut [MaybeUninit<<EValuePtrListElem as Storable>::__Storage>],
            >(storage)
        };
        let mut storage_iter = storage.iter_mut();
        loop {
            match (values.next(), storage_iter.next()) {
                (Some(value), Some(storage)) => {
                    storage.write(value.map(|value| value.cpp()).unwrap_or(et_c::EValueRef {
                        ptr: std::ptr::null(),
                    }));
                }
                (None, None) => break,
                _ => panic!("Mismatched lengths"),
            }
        }
        // Safety: We wrote to all elements of the slice.
        let storage = unsafe {
            std::mem::transmute::<
                &mut [MaybeUninit<<EValuePtrListElem as Storable>::__Storage>],
                &mut [<EValuePtrListElem as Storable>::__Storage],
            >(storage)
        };
        Self(EValuePtrListInner::Slice((storage, PhantomData)))
    }

    /// Create a new list with the give values using the given storage.
    ///
    /// Usually such list is used as an input to a [`BoxedEvalueList`]. In that case the
    /// values should be of the same type.
    ///
    /// This function does not allocate on the heap.
    ///
    /// # Panics
    ///
    /// Panics if the length of the `values` and `storage` do not match.
    pub fn new_in_storage(
        values: impl IntoIterator<Item = &'a EValue<'a>>,
        storage: Pin<&'a mut [Storage<EValuePtrListElem>]>,
    ) -> Self {
        Self::new_in_storage_impl(values.into_iter().map(Some), storage)
    }

    /// Create a new list with the give values using the given storage, where some values can be `None`.
    ///
    /// Usually such list is used as an input to a [`BoxedEvalueList`]. In that case the
    /// values should be of the same type, and only `Option<TensorAny>` support `None` values.
    ///
    /// This function does not allocate on the heap.
    ///
    /// # Panics
    ///
    /// Panics if the length of the `values` and `storage` do not match.
    pub fn new_optional_in_storage(
        values: impl IntoIterator<Item = Option<&'a EValue<'a>>>,
        storage: Pin<&'a mut [Storage<EValuePtrListElem>]>,
    ) -> Self {
        Self::new_in_storage_impl(values, storage)
    }

    fn as_slice(&self) -> &[et_c::EValueRef] {
        match &self.0 {
            #[cfg(feature = "alloc")]
            EValuePtrListInner::Vec((values, _)) => values.as_slice(),
            EValuePtrListInner::Slice((values, _)) => values,
        }
    }

    /// Returns None if index is out of range.
    /// Returns Some(None) if the pointer at the given entry is null.
    fn get(&self, index: usize) -> Option<Option<EValue<'_>>> {
        let ptr = *self.as_slice().get(index)?;
        Some(if ptr.ptr.is_null() {
            None
        } else {
            Some(unsafe { EValue::from_inner_ref(ptr) })
        })
    }
}
/// An element within a [`EValuePtrList`].
///
/// Used solely for the `Storable` implementation:
/// ```rust,ignore
/// let (evalue1, evalue2, evalue3) = (EValue::new(42), EValue::new(17), EValue::new(6));
/// let wrapped_vals_storage = executorch::storage!(EValuePtrListElement, [3]);
/// let wrapped_vals = EValuePtrList::new_in_storage([&evalue1, &evalue2, &evalue3], wrapped_vals_storage);
/// ```
pub struct EValuePtrListElem(#[allow(unused)] et_c::EValueRef);
impl Storable for EValuePtrListElem {
    type __Storage = et_c::EValueRef;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evalue::EValue;
    use crate::storage;

    #[test]
    #[should_panic]
    fn evalue_ptr_list_length_mismatch() {
        let evalue1_storage = storage!(EValue);
        let evalue2_storage = storage!(EValue);
        let evalue3_storage = storage!(EValue);
        let evalue1 = EValue::new_in_storage(42, evalue1_storage);
        let evalue2 = EValue::new_in_storage(17, evalue2_storage);
        let evalue3 = EValue::new_in_storage(6, evalue3_storage);

        let wrapped_vals_storage = storage!(EValuePtrListElem, [2]); // length mismatch
        let _ = EValuePtrList::new_in_storage([&evalue1, &evalue2, &evalue3], wrapped_vals_storage);
    }

    #[test]
    fn length_mismatch() {
        let evalue1_storage = storage!(EValue);
        let evalue2_storage = storage!(EValue);
        let evalue3_storage = storage!(EValue);
        let evalue1 = EValue::new_in_storage(42, evalue1_storage);
        let evalue2 = EValue::new_in_storage(17, evalue2_storage);
        let evalue3 = EValue::new_in_storage(6, evalue3_storage);

        let wrapped_vals_storage = storage!(EValuePtrListElem, [3]);
        let wrapped_vals =
            EValuePtrList::new_in_storage([&evalue1, &evalue2, &evalue3], wrapped_vals_storage);
        let unwrapped_vals = storage!(i64, [2]); // length mismatch

        let res = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals);
        assert!(matches!(res, Err(Error::CError(CError::InvalidArgument))));
    }

    #[test]
    fn wrong_type() {
        let evalue1_storage = storage!(EValue);
        let evalue2_storage = storage!(EValue);
        let evalue3_storage = storage!(EValue);
        let evalue1 = EValue::new_in_storage(42, evalue1_storage);
        let evalue2 = EValue::new_in_storage(17, evalue2_storage);
        let evalue3 = EValue::new_in_storage(6.5, evalue3_storage); // wrong type

        let wrapped_vals_storage = storage!(EValuePtrListElem, [3]);
        let wrapped_vals =
            EValuePtrList::new_in_storage([&evalue1, &evalue2, &evalue3], wrapped_vals_storage);
        let unwrapped_vals = storage!(i64, [3]);

        let res = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals);
        assert!(matches!(res, Err(Error::CError(CError::InvalidType))));
    }

    #[test]
    fn null_element() {
        let evalue1_storage = storage!(EValue);
        let evalue2_storage = storage!(EValue);
        // let evalue3_storage = None;
        let evalue1 = EValue::new_in_storage(42, evalue1_storage);
        let evalue2 = EValue::new_in_storage(17, evalue2_storage);
        let evalue3 = None; // null element

        let wrapped_vals_storage = storage!(EValuePtrListElem, [3]);
        let wrapped_vals = EValuePtrList::new_optional_in_storage(
            [Some(&evalue1), Some(&evalue2), evalue3],
            wrapped_vals_storage,
        );
        let unwrapped_vals = storage!(i64, [3]);

        let res = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals);
        assert!(matches!(res, Err(Error::CError(CError::InvalidType))));
    }
}
