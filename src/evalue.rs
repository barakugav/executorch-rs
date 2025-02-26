//! Module for [`EValue`] and related types.
//!
//! [`EValue`] is a type-erased value that can hold different types like scalars, lists or tensors. It is used to pass
//! arguments to and return values from the runtime.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::pin::Pin;

use crate::memory::{Storable, Storage};
use crate::tensor::{self, TensorAny, TensorBase};
use crate::util::{ArrayRef, Destroy, NonTriviallyMovable, __ArrayRefImpl};
use crate::{et_c, et_rs_c, CError, Error, Result};

/// A tag indicating the type of the value stored in an [`EValue`].
///
/// - `Tensor`: Tag for value [`TensorAny`].
/// - `String`: Tag for value `&[c_char]`.
/// - `Double`: Tag for value `f64`.
/// - `Int`: Tag for value `i64`.
/// - `Bool`: Tag for value `bool`.
/// - `ListBool`: Tag for value `&[bool]`.
/// - `ListDouble`: Tag for value `&[f64]`.
/// - `ListInt`: Tag for value `&[i64]`.
/// - `ListTensor`: Tag for value `&[TensorAny]`.
/// - `ListScalar`: unsupported at the moment.
/// - `ListOptionalTensor`: Tag for value `&[Option<TensorAny>]`.
///
pub use et_c::runtime::Tag;

/// Aggregate typing system similar to IValue only slimmed down with less
/// functionality, no dependencies on atomic, and fewer supported types to better
/// suit embedded systems (ie no intrusive ptr)
pub struct EValue<'a>(NonTriviallyMovable<'a, et_rs_c::EValue>);
impl<'a> EValue<'a> {
    /// Create a new [`EValue`] on the heap.
    ///
    /// # Arguments
    ///
    /// * `init` - A closure that initializes the value. This is intended to be a call to a Cpp function that constructs
    ///     the value.
    ///
    /// # Safety
    ///
    /// The closure must initialize the value correctly, otherwise the value will be in an invalid state.
    #[cfg(feature = "alloc")]
    unsafe fn new_impl(init: impl FnOnce(*mut et_rs_c::EValue)) -> Self {
        Self(unsafe { NonTriviallyMovable::new_boxed(init) })
    }

    /// Create a new [`EValue`] from a value that can be converted into one.
    ///
    /// The underlying Cpp object is allocated on the heap, which is preferred on systems in which allocations are
    /// available.
    /// For an identical version that allocates the object in a given storage (possibly on the stack), see the
    /// [`new_in_storage`][EValue::new_in_storage] method.
    /// Note that the inner data is not copied, and the required allocation is small.
    #[cfg(feature = "alloc")]
    pub fn new(value: impl IntoEValue<'a>) -> Self {
        value.into_evalue()
    }

    /// Create a new [`EValue`] in the given storage.
    ///
    /// # Arguments
    ///
    /// * `init` - A closure that initializes the value. This is intended to be a call to a Cpp function that constructs
    ///     the value.
    /// * `storage` - The storage in which the value will be allocated.
    ///
    /// # Safety
    ///
    /// The closure must initialize the value correctly, otherwise the value will be in an invalid state.
    unsafe fn new_in_storage_impl(
        init: impl FnOnce(*mut et_rs_c::EValue),
        storage: Pin<&'a mut Storage<EValue>>,
    ) -> Self {
        Self(unsafe { NonTriviallyMovable::new_in_storage(init, storage) })
    }

    /// Create a new [`EValue`] from a value that can be converted into one in the given storage.
    ///
    /// This function is identical to [`EValue::new`][EValue::new], but it allows to create the evalue without the
    /// use of a heap.
    /// Few examples of ways to create an [`EValue`]:
    /// ```rust,ignore
    /// // The value is allocated on the heap
    /// let evalue = EValue::new(value);
    ///
    /// // The value is allocated on the stack
    /// let storage = executorch::storage!(EValue);
    /// let evalue = EValue::new_in_storage(value, storage);
    ///
    /// // The value is allocated using a memory allocator
    /// let allocator: impl AsRef<MemoryAllocator> = ...; // usually global
    /// let evalue = EValue::new_in_storage(value, allocator.as_ref().allocate_pinned().unwrap());
    /// ```
    /// Note that the inner data is not copied, and the required allocation is small.
    /// See [`Storage`] for more information.
    pub fn new_in_storage(
        value: impl IntoEValue<'a>,
        storage: Pin<&'a mut Storage<EValue>>,
    ) -> Self {
        value.into_evalue_in_storage(storage)
    }

    pub(crate) fn from_inner_ref(value: &'a et_rs_c::EValue) -> Self {
        Self(NonTriviallyMovable::from_ref(value))
    }

    /// Create a new [`EValue`] by moving from an existing [`EValue`].
    ///
    /// # Safety
    ///
    /// The given value should not be used after this function is called, and its Cpp destructor should be called.
    #[cfg(feature = "alloc")]
    #[allow(dead_code)]
    pub(crate) unsafe fn move_from(value: &mut et_rs_c::EValue) -> Self {
        Self(unsafe { NonTriviallyMovable::new_boxed(|p| et_rs_c::EValue_move(value, p)) })
    }

    pub(crate) fn as_evalue(&self) -> &et_rs_c::EValue {
        self.0.as_ref()
    }

    /// Create a new [`EValue`] with the no value (tag `None`).
    #[cfg(feature = "alloc")]
    pub fn none() -> Self {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::executorch_EValue_new_none(p)) }
    }

    /// Create a new [`EValue`] with the no value (tag `None`) in the given storage.
    pub fn none_in_storage(storage: Pin<&'a mut Storage<EValue>>) -> Self {
        // Safety: the closure init the pointer
        unsafe { EValue::new_in_storage_impl(|p| et_rs_c::executorch_EValue_new_none(p), storage) }
    }

    /// Check if the value is of type `None`.
    pub fn is_none(&self) -> bool {
        self.tag() == Tag::None
    }

    /// Get a reference to the value as an `i64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_i64(&self) -> i64 {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[i64]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_i64_list(&self) -> &[i64] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as an `f64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_f64(&self) -> f64 {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[f64]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_f64_list(&self) -> &[f64] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `bool`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_bool(&self) -> bool {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[bool]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_bool_list(&self) -> &[bool] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[c_char]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_chars(&self) -> &[std::ffi::c_char] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `CStr`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    pub fn as_cstr(&self) -> &CStr {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a [`TensorAny`].
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_tensor(&self) -> TensorAny {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a [`TensorList`].
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    pub fn as_tensor_list(&self) -> TensorList {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a [`OptionalTensorList`].
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    pub fn as_optional_tensor_list(&self) -> OptionalTensorList {
        self.try_into().expect("Invalid type")
    }

    /// Get the tag indicating the type of the value.
    ///
    /// Returns `None` if the inner Cpp tag is `None`.
    pub fn tag(&self) -> Tag {
        unsafe { et_rs_c::executorch_EValue_tag(self.as_evalue()) }
    }
}
impl Destroy for et_rs_c::EValue {
    unsafe fn destroy(&mut self) {
        unsafe { et_rs_c::EValue_destructor(self) }
    }
}

impl Storable for EValue<'_> {
    type __Storage = et_rs_c::EValue;
}

/// A type that can be converted into an [`EValue`].
pub trait IntoEValue<'a> {
    /// Convert the value into an [`EValue`], with an allocation on the heap.
    ///
    /// This is the preferred method to create an [`EValue`] when allocations are available.
    /// Use `into_evalue_in_storage` for an identical version that allow to allocate the object without the
    /// use of a heap.
    /// Note that the inner data is not copied, and the required allocation is small.
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a>;

    /// Convert the value into an [`EValue`], using the given storage.
    ///
    /// This function is identical to `into_evalue`, but it allows to create the evalue without the
    /// use of a heap.
    /// See [`Storage`] for more information.
    /// Note that the inner data is not copied, and the required allocation is small.
    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a>;
}
impl<'a> IntoEValue<'a> for i64 {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_i64(p, self)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_i64(p, self), storage) }
    }
}
impl<'a> IntoEValue<'a> for BoxedEvalueList<'a, i64> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_i64_list(p, self.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_i64_list(p, self.0), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for f64 {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_f64(p, self)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_f64(p, self), storage) }
    }
}
impl<'a> IntoEValue<'a> for &'a [f64] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_f64_list(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_f64_list(p, arr.0), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for bool {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_bool(p, self)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_bool(p, self), storage) }
    }
}
impl<'a> IntoEValue<'a> for &'a [bool] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_bool_list(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_bool_list(p, arr.0), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for &'a [std::ffi::c_char] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_string(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_string(p, arr.0), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for &'a CStr {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        crate::util::cstr2chars(self).into_evalue()
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        crate::util::cstr2chars(self).into_evalue_in_storage(storage)
    }
}
impl<'a, D: tensor::Data> IntoEValue<'a> for TensorBase<'a, D> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor())) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor()),
                storage,
            )
        }
    }
}
impl<'a, D: tensor::Data> IntoEValue<'a> for &'a TensorBase<'_, D> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor())) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor()),
                storage,
            )
        }
    }
}
#[cfg(feature = "tensor-ptr")]
impl<'a, D: tensor::Data> IntoEValue<'a> for &'a tensor::TensorPtr<'_, D> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        self.as_tensor().into_evalue()
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        self.as_tensor().into_evalue_in_storage(storage)
    }
}
impl<'a> IntoEValue<'a> for BoxedEvalueList<'a, TensorAny<'a>> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_tensor_list(p, self.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_rs_c::EValue_new_from_tensor_list(p, self.0),
                storage,
            )
        }
    }
}
impl<'a> IntoEValue<'a> for BoxedEvalueList<'a, Option<TensorAny<'a>>> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_optional_tensor_list(p, self.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_rs_c::EValue_new_from_optional_tensor_list(p, self.0),
                storage,
            )
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a, T> From<T> for EValue<'a>
where
    T: IntoEValue<'a>,
{
    fn from(val: T) -> Self {
        val.into_evalue()
    }
}

impl TryFrom<&EValue<'_>> for i64 {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<Self> {
        if value.tag() == Tag::Int {
            Ok(unsafe { et_rs_c::EValue_as_i64(value.as_evalue()) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [i64] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListInt {
            Ok(unsafe { et_rs_c::EValue_as_i64_list(value.as_evalue()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl TryFrom<&EValue<'_>> for f64 {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<Self> {
        if value.tag() == Tag::Double {
            Ok(unsafe { et_rs_c::EValue_as_f64(value.as_evalue()) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [f64] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListDouble {
            Ok(unsafe { et_rs_c::EValue_as_f64_list(value.as_evalue()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl TryFrom<&EValue<'_>> for bool {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<Self> {
        if value.tag() == Tag::Bool {
            Ok(unsafe { et_rs_c::EValue_as_bool(value.as_evalue()) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [bool] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListBool {
            Ok(unsafe { et_rs_c::EValue_as_bool_list(value.as_evalue()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [std::ffi::c_char] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::String {
            Ok(unsafe { et_rs_c::EValue_as_string(value.as_evalue()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a CStr {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        let chars: &[std::ffi::c_char] = value.try_into()?;
        Ok(unsafe { CStr::from_ptr(chars.as_ptr()) })
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for TensorAny<'a> {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::Tensor {
            let tensor = unsafe { et_rs_c::EValue_as_tensor(value.as_evalue()) };
            Ok(unsafe { TensorAny::from_inner_ref(&*tensor) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
// impl<'a> TryFrom<EValue<'a>> for Tensor<'a> {
//     type Error = Error;
//     fn try_from(mut value: EValue<'a>) -> Result<Tensor<'a>> {
//         match value.tag() {
//             Some(Tag::Tensor) => Ok(unsafe {
//                 value.0.tag = et_c::Tag::None;
//                 let inner = ManuallyDrop::take(&mut value.0.payload.as_tensor);
//                 Tensor::from_inner(inner)
//             }),
//             _ => Err(Error::CError(CError::InvalidType)),
//         }
//     }
// }
impl<'a> TryFrom<&'a EValue<'_>> for TensorList<'a> {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListTensor {
            let list = unsafe { et_rs_c::EValue_as_tensor_list(value.as_evalue()) };
            Ok(unsafe { Self::from_array_ref(list) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for OptionalTensorList<'a> {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListOptionalTensor {
            let list = unsafe { et_rs_c::EValue_as_optional_tensor_list(value.as_evalue()) };
            Ok(unsafe { Self::from_array_ref(list) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
#[cfg(feature = "ndarray")]
impl std::fmt::Debug for EValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut st = f.debug_struct("EValue");
        st.field("tag", &self.tag());
        match self.tag() {
            Tag::Int => st.field("value", &self.as_i64()),
            Tag::Double => st.field("value", &self.as_f64()),
            Tag::Bool => st.field("value", &self.as_bool()),
            Tag::Tensor => st.field("value", &self.as_tensor()),
            Tag::String => st.field("value", &self.as_chars()),
            Tag::ListInt => st.field("value", &self.as_i64_list()),
            Tag::ListDouble => st.field("value", &self.as_f64_list()),
            Tag::ListBool => st.field("value", &self.as_bool_list()),
            Tag::ListTensor => st.field("value", &self.as_tensor_list()),
            Tag::ListOptionalTensor => st.field("value", &self.as_optional_tensor_list()),
            Tag::ListScalar => st.field("value", &"Unsupported type: ListScalar"),
            Tag::None => st.field("value", &"None"),
        };
        st.finish()
    }
}

/// A list of tensors.
pub struct TensorList<'a>(&'a [et_c::aten::Tensor]);
impl TensorList<'_> {
    /// Safety: the array must be valid for the lifetime of the returned list.
    unsafe fn from_array_ref(array: et_rs_c::ArrayRefTensor) -> Self {
        Self(unsafe { std::slice::from_raw_parts(array.data, array.len) })
    }

    /// Get the length of the list.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the tensor at the given index.
    pub fn get(&self, index: usize) -> Option<TensorAny> {
        self.0.get(index).map(TensorAny::from_inner_ref)
    }
}
#[cfg(feature = "ndarray")]
impl std::fmt::Debug for TensorList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut l = f.debug_list();
        for i in 0..self.len() {
            l.entry(&self.get(i).unwrap());
        }
        l.finish()
    }
}

/// A list of optional tensors.
pub struct OptionalTensorList<'a>(&'a [et_rs_c::OptionalTensor]);
impl OptionalTensorList<'_> {
    /// Safety: the array must be valid for the lifetime of the returned list.
    unsafe fn from_array_ref(array: et_rs_c::ArrayRefOptionalTensor) -> Self {
        Self(unsafe { std::slice::from_raw_parts(array.data, array.len) })
    }

    /// Get the length of the list.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the tensor at the given index.
    ///
    /// # Returns
    ///
    /// - `None` if the index is out of bounds.
    /// - `Some(None)` if the tensor at the index is `None`.
    /// - `Some(Some(tensor))` if the tensor at the index is not `None`.
    pub fn get(&self, index: usize) -> Option<Option<TensorAny>> {
        self.0.get(index).map(|opt| {
            opt.init_
                .then(|| TensorAny::from_inner_ref(unsafe { &opt.storage_.value_ }))
        })
    }
}
#[cfg(feature = "ndarray")]
impl std::fmt::Debug for OptionalTensorList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut l = f.debug_list();
        for i in 0..self.len() {
            l.entry(&self.get(i).unwrap());
        }
        l.finish()
    }
}

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
///     list, boxed in `EValue`.
/// - `unwrapped_vals`: a list of `[T]`, initially uninitialized but when a reference
///     to the actual `T` values is required the values are "unwrapped" from the boxed
///     `EValue` into this array, and returned to the user as a slice.
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
    ///     within the `EValue` must match the type `T`.
    /// - `unwrapped_vals`: an allocation of the unwrapped values. The length of the allocation must
    ///    match the length of the `wrapped_vals`.
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

        let wrapped_vals = et_rs_c::ArrayRefEValuePtr {
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
        wrapped_vals: et_rs_c::ArrayRefEValuePtr,
        unwrapped_vals: Pin<&mut [Storage<Self::Element<'_>>]>,
    ) -> Result<Self>
    where
        Self: Sized;

    private_decl! {}
}

macro_rules! impl_boxed_evalue_list {
    ($element:path, $list_impl:path, $unwrapped_span_type:path, $element_tag:ident, $allow_null_element:expr) => {
        impl<'a> BoxedEvalueListElement<'a> for $element {
            const __ELEMENT_TAG: Tag = Tag::$element_tag;
            const __ALLOW_NULL_ELEMENT: bool = $allow_null_element;
            type __ListImpl = $list_impl;
            private_impl! {}
        }
        impl __BoxedEvalueListImpl for $list_impl {
            type Element<'a> = $element;

            unsafe fn __new(
                wrapped_vals: et_rs_c::ArrayRefEValuePtr,
                unwrapped_vals: Pin<&mut [Storage<Self::Element<'_>>]>,
            ) -> Result<Self> {
                // Safety: we dont move out of the pinned slice.
                let unwrapped_vals = unsafe { unwrapped_vals.get_unchecked_mut() };
                Ok(Self {
                    wrapped_vals,
                    unwrapped_vals: {
                        $unwrapped_span_type {
                            data: unwrapped_vals.as_mut_ptr()
                                as *mut <Self::Element<'_> as Storable>::__Storage,
                            len: unwrapped_vals.len(),
                        }
                    },
                })
            }

            private_impl! {}
        }
    };
}
impl_boxed_evalue_list!(
    i64,
    et_rs_c::BoxedEvalueListI64,
    et_rs_c::SpanI64,
    Int,
    false
);
impl_boxed_evalue_list!(
    TensorAny<'a>,
    et_rs_c::BoxedEvalueListTensor,
    et_rs_c::SpanTensor,
    Tensor,
    false
);
impl_boxed_evalue_list!(
    Option<TensorAny<'a>>,
    et_rs_c::BoxedEvalueListOptionalTensor,
    et_rs_c::SpanOptionalTensor,
    Tensor,
    true
);

/// A list of pointers to `EValue`.
///
/// Usually such list is used as an input to a [`BoxedEvalueList`].
pub struct EValuePtrList<'a>(EValuePtrListInner<'a>);
enum EValuePtrListInner<'a> {
    #[cfg(feature = "alloc")]
    Vec(
        (
            crate::alloc::Vec<*const et_rs_c::EValue>,
            // A lifetime for the `*const EValue` values
            PhantomData<&'a ()>,
        ),
    ),
    Slice(
        (
            &'a [*const et_rs_c::EValue],
            // A lifetime for the `*const EValue` values
            PhantomData<&'a ()>,
        ),
    ),
}
impl<'a> EValuePtrList<'a> {
    #[cfg(feature = "alloc")]
    fn new_impl(values: impl IntoIterator<Item = Option<&'a EValue<'a>>>) -> Self {
        let values: crate::alloc::Vec<*const et_rs_c::EValue> = values
            .into_iter()
            .map(|value| match value {
                Some(value) => value.as_evalue() as *const _,
                None => std::ptr::null(),
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
                    storage.write(match value {
                        Some(value) => value.as_evalue() as *const _,
                        None => std::ptr::null(),
                    });
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

    fn as_slice(&self) -> &[*const et_rs_c::EValue] {
        match &self.0 {
            #[cfg(feature = "alloc")]
            EValuePtrListInner::Vec((values, _)) => values.as_slice(),
            EValuePtrListInner::Slice((values, _)) => values,
        }
    }

    /// Returns None if index is out of range.
    /// Returns Some(None) if the pointer at the given entry is null.
    fn get(&self, index: usize) -> Option<Option<EValue>> {
        let ptr = *self.as_slice().get(index)?;
        Some(if ptr.is_null() {
            None
        } else {
            Some(unsafe { EValue::from_inner_ref(&*ptr) })
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
pub struct EValuePtrListElem(#[allow(dead_code)] *const et_rs_c::EValue);
impl Storable for EValuePtrListElem {
    type __Storage = *const et_rs_c::EValue;
}

#[cfg(test)]
mod tests {
    use crate::storage;
    #[cfg(feature = "tensor-ptr")]
    use crate::tensor::TensorPtr;
    use crate::tensor::{SizesType, Tensor, TensorImpl};

    use super::*;

    #[test]
    fn none() {
        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::none();
            assert_eq!(evalue.tag(), Tag::None);
            assert!(evalue.is_none());
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::none_in_storage(storage);
            assert_eq!(evalue.tag(), Tag::None);
            assert!(evalue.is_none());
        }
    }

    #[test]
    fn i64() {
        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(42);
            assert_eq!(evalue.tag(), Tag::Int);
            assert_eq!(evalue.as_i64(), 42);
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::new_in_storage(17, storage);
            assert_eq!(evalue.tag(), Tag::Int);
            assert_eq!(evalue.as_i64(), 17);
        }
    }

    #[test]
    fn i64_list() {
        #[cfg(feature = "alloc")]
        {
            let (evalue1, evalue2, evalue3) = (EValue::new(42), EValue::new(17), EValue::new(6));
            let wrapped_vals = EValuePtrList::new([&evalue1, &evalue2, &evalue3]);
            let mut unwrapped_vals = storage!(i64, (3));
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals.as_mut()).unwrap();

            let evalue = EValue::new(list);
            assert_eq!(evalue.tag(), Tag::ListInt);
            assert_eq!(evalue.as_i64_list(), &[42, 17, 6]);
        }
        {
            let evalue1_storage = storage!(EValue);
            let evalue2_storage = storage!(EValue);
            let evalue3_storage = storage!(EValue);
            let evalue1 = EValue::new_in_storage(42, evalue1_storage);
            let evalue2 = EValue::new_in_storage(17, evalue2_storage);
            let evalue3 = EValue::new_in_storage(6, evalue3_storage);

            let wrapped_vals_storage = storage!(EValuePtrListElem, [3]);
            let wrapped_vals =
                EValuePtrList::new_in_storage([&evalue1, &evalue2, &evalue3], wrapped_vals_storage);
            let unwrapped_vals = storage!(i64, [3]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue_storage = storage!(EValue);
            let evalue = EValue::new_in_storage(list, evalue_storage);
            assert_eq!(evalue.tag(), Tag::ListInt);
            assert_eq!(evalue.as_i64_list(), &[42, 17, 6]);
        }
    }

    #[test]
    fn f64() {
        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(42.0);
            assert_eq!(evalue.tag(), Tag::Double);
            assert_eq!(evalue.as_f64(), 42.0);
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::new_in_storage(17.0, storage);
            assert_eq!(evalue.tag(), Tag::Double);
            assert_eq!(evalue.as_f64(), 17.0);
        }
    }

    #[test]
    fn f64_list() {
        let list = [42.0, 17.0, 6.0];

        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(list.as_slice());
            assert_eq!(evalue.tag(), Tag::ListDouble);
            assert_eq!(evalue.as_f64_list(), [42.0, 17.0, 6.0]);
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::new_in_storage(list.as_slice(), storage);
            assert_eq!(evalue.tag(), Tag::ListDouble);
            assert_eq!(evalue.as_f64_list(), [42.0, 17.0, 6.0]);
        }
    }

    #[test]
    fn bool() {
        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(true);
            assert_eq!(evalue.tag(), Tag::Bool);
            assert!(evalue.as_bool());
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::new_in_storage(false, storage);
            assert_eq!(evalue.tag(), Tag::Bool);
            assert!(!evalue.as_bool());
        }
    }

    #[test]
    fn bool_list() {
        let list = [true, false, true];

        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(list.as_slice());
            assert_eq!(evalue.tag(), Tag::ListBool);
            assert_eq!(evalue.as_bool_list(), [true, false, true]);
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::new_in_storage(list.as_slice(), storage);
            assert_eq!(evalue.tag(), Tag::ListBool);
            assert_eq!(evalue.as_bool_list(), [true, false, true]);
        }
    }

    #[test]
    fn string() {
        let string = cstr::cstr!(b"hello world!");
        let chars = crate::util::cstr2chars(string);

        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(string);
            assert_eq!(evalue.tag(), Tag::String);
            assert_eq!(evalue.as_cstr(), string);
            assert_eq!(evalue.as_chars(), chars);
        }
        {
            let storage = storage!(EValue);
            let evalue = EValue::new_in_storage(string, storage);
            assert_eq!(evalue.tag(), Tag::String);
            assert_eq!(evalue.as_cstr(), string);
            assert_eq!(evalue.as_chars(), chars);
        }
    }

    #[test]
    fn tensor() {
        let data: [i32; 3] = [42, 17, 6];

        #[cfg(feature = "alloc")]
        {
            let sizes = [data.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides);
            let tensor = Tensor::new(&tensor_impl);

            // Borrow tensor by EValue
            let evalue = EValue::new(&tensor);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);

            // Move tensor into evalue
            let evalue = EValue::new(tensor);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);
        }
        #[cfg(feature = "tensor-ptr")]
        {
            let tensor = TensorPtr::from_slice(&data);
            let evalue = EValue::new(&tensor);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);
        }
        {
            let sizes = [data.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            // Borrow tensor by EValue
            let evalue_storage = storage!(EValue);
            let evalue = EValue::new_in_storage(&tensor, evalue_storage);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);

            // Move tensor into evalue
            let evalue_storage = storage!(EValue);
            let evalue = EValue::new_in_storage(tensor, evalue_storage);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);
        }
    }

    #[test]
    fn tensor_list() {
        let data1: [i32; 3] = [42, 17, 6];
        let data2: [i32; 2] = [55, 8];
        let data3: [i32; 2] = [106, 144];

        #[cfg(feature = "alloc")]
        {
            let sizes = [data1.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides);
            let tensor1 = Tensor::new(&tensor_impl);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides);
            let tensor2 = Tensor::new(&tensor_impl);

            let sizes = [data3.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data3, &dim_order, &strides);
            let tensor3 = Tensor::new(&tensor_impl);

            let evalue1 = EValue::new(tensor1);
            let evalue2 = EValue::new(tensor2);
            let evalue3 = EValue::new(tensor3);
            let wrapped_vals = EValuePtrList::new([&evalue1, &evalue2, &evalue3]);
            let unwrapped_vals = storage!(TensorAny, [3]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue = EValue::new(list);
            assert_eq!(evalue.tag(), Tag::ListTensor);
            let tensor_list = evalue.as_tensor_list();
            assert_eq!(tensor_list.len(), 3);

            for (i, data) in [data1.as_slice(), &data2, &data3].iter().enumerate() {
                let tensor = tensor_list.get(i).unwrap().into_typed::<i32>();
                let tensor_data =
                    unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
                assert_eq!(&tensor_data, data);
            }
        }
        #[cfg(feature = "tensor-ptr")]
        {
            let tensor1 = TensorPtr::from_slice(&data1);
            let tensor2 = TensorPtr::from_slice(&data2);
            let tensor3 = TensorPtr::from_slice(&data3);
            let evalue1 = EValue::new(&tensor1);
            let evalue2 = EValue::new(&tensor2);
            let evalue3 = EValue::new(&tensor3);
            let wrapped_vals = EValuePtrList::new([&evalue1, &evalue2, &evalue3]);
            let unwrapped_vals = storage!(TensorAny, [3]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue = EValue::new(list);
            assert_eq!(evalue.tag(), Tag::ListTensor);
            let tensor_list = evalue.as_tensor_list();
            assert_eq!(tensor_list.len(), 3);

            for (i, data) in [data1.as_slice(), &data2, &data3].iter().enumerate() {
                let tensor = tensor_list.get(i).unwrap().into_typed::<i32>();
                let tensor_data =
                    unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
                assert_eq!(&tensor_data, data);
            }
        }
        {
            let sizes = [data1.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor1 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor2 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let sizes = [data3.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data3, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor3 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let evalue_storage = storage!(EValue);
            let evalue1 = EValue::new_in_storage(&tensor1, evalue_storage);
            let evalue_storage = storage!(EValue);
            let evalue2 = EValue::new_in_storage(&tensor2, evalue_storage);
            let evalue_storage = storage!(EValue);
            let evalue3 = EValue::new_in_storage(&tensor3, evalue_storage);

            let wrapped_vals_storage = storage!(EValuePtrListElem, [3]);
            let wrapped_vals =
                EValuePtrList::new_in_storage([&evalue1, &evalue2, &evalue3], wrapped_vals_storage);
            let unwrapped_vals = storage!(TensorAny, [3]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue_storage = storage!(EValue);
            let evalue = EValue::new_in_storage(list, evalue_storage);
            assert_eq!(evalue.tag(), Tag::ListTensor);
            let tensor_list = evalue.as_tensor_list();
            assert_eq!(tensor_list.len(), 3);

            for (i, data) in [data1.as_slice(), &data2, &data3].iter().enumerate() {
                let tensor = tensor_list.get(i).unwrap().into_typed::<i32>();
                let tensor_data =
                    unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
                assert_eq!(&tensor_data, data);
            }
        }
    }

    #[test]
    fn optional_tensor_list() {
        let data1: [i32; 3] = [42, 17, 6];
        let data2: [i32; 2] = [55, 8];
        let data4: [i32; 2] = [106, 144];

        #[cfg(feature = "alloc")]
        {
            let sizes = [data1.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides);
            let tensor1 = Tensor::new(&tensor_impl);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides);
            let tensor2 = Tensor::new(&tensor_impl);

            // let tensor3 = None;

            let sizes = [data4.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data4, &dim_order, &strides);
            let tensor4 = Tensor::new(&tensor_impl);

            let evalue1 = EValue::new(tensor1);
            let evalue2 = EValue::new(tensor2);
            let evalue3 = None;
            let evalue4 = EValue::new(tensor4);
            let wrapped_vals = EValuePtrList::new_optional([
                Some(&evalue1),
                Some(&evalue2),
                evalue3,
                Some(&evalue4),
            ]);
            let unwrapped_vals = storage!(Option<TensorAny>, [4]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue = EValue::new(list);
            assert_eq!(evalue.tag(), Tag::ListOptionalTensor);
            let tensor_list = evalue.as_optional_tensor_list();
            assert_eq!(tensor_list.len(), 4);

            for (i, data) in [Some(data1.as_slice()), Some(&data2), None, Some(&data4)]
                .iter()
                .enumerate()
            {
                let tensor = tensor_list.get(i).unwrap();
                assert_eq!(tensor.is_some(), data.is_some());
                match (tensor, data) {
                    (None, None) => {}
                    (Some(tensor), Some(data)) => {
                        let tensor_data = unsafe {
                            std::slice::from_raw_parts(
                                tensor.into_typed::<i32>().as_ptr(),
                                data.len(),
                            )
                        };
                        assert_eq!(&tensor_data, data);
                    }
                    _ => unreachable!(),
                }
            }
        }
        #[cfg(feature = "tensor-ptr")]
        {
            let tensor1 = TensorPtr::from_slice(&data1);
            let tensor2 = TensorPtr::from_slice(&data2);
            // let tensor3 = None;
            let tensor4 = TensorPtr::from_slice(&data4);
            let evalue1 = EValue::new(&tensor1);
            let evalue2 = EValue::new(&tensor2);
            let evalue3 = None;
            let evalue4 = EValue::new(&tensor4);
            let wrapped_vals = EValuePtrList::new_optional([
                Some(&evalue1),
                Some(&evalue2),
                evalue3,
                Some(&evalue4),
            ]);
            let unwrapped_vals = storage!(Option<TensorAny>, [4]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue = EValue::new(list);
            assert_eq!(evalue.tag(), Tag::ListOptionalTensor);
            let tensor_list = evalue.as_optional_tensor_list();
            assert_eq!(tensor_list.len(), 4);

            for (i, data) in [Some(data1.as_slice()), Some(&data2), None, Some(&data4)]
                .iter()
                .enumerate()
            {
                let tensor = tensor_list.get(i).unwrap();
                assert_eq!(tensor.is_some(), data.is_some());
                match (tensor, data) {
                    (None, None) => {}
                    (Some(tensor), Some(data)) => {
                        let tensor_data = unsafe {
                            std::slice::from_raw_parts(
                                tensor.into_typed::<i32>().as_ptr(),
                                data.len(),
                            )
                        };
                        assert_eq!(&tensor_data, data);
                    }
                    _ => unreachable!(),
                }
            }
        }
        {
            let sizes = [data1.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor1 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor2 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            // let tensor3 = None;

            let sizes = [data4.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data4, &dim_order, &strides);
            let tensor_storage = storage!(Tensor<i32>);
            let tensor4 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let evalue_storage = storage!(EValue);
            let evalue1 = EValue::new_in_storage(&tensor1, evalue_storage);
            let evalue_storage = storage!(EValue);
            let evalue2 = EValue::new_in_storage(&tensor2, evalue_storage);
            let evalue3 = None;
            let evalue_storage = storage!(EValue);
            let evalue4 = EValue::new_in_storage(&tensor4, evalue_storage);

            let wrapped_vals_storage = storage!(EValuePtrListElem, [4]);
            let wrapped_vals = EValuePtrList::new_optional_in_storage(
                [Some(&evalue1), Some(&evalue2), evalue3, Some(&evalue4)],
                wrapped_vals_storage,
            );
            let unwrapped_vals = storage!(Option<TensorAny>, [4]);
            let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();

            let evalue_storage = storage!(EValue);
            let evalue = EValue::new_in_storage(list, evalue_storage);
            assert_eq!(evalue.tag(), Tag::ListOptionalTensor);
            let tensor_list = evalue.as_optional_tensor_list();
            assert_eq!(tensor_list.len(), 4);

            for (i, data) in [Some(data1.as_slice()), Some(&data2), None, Some(&data4)]
                .iter()
                .enumerate()
            {
                let tensor = tensor_list.get(i).unwrap();
                assert_eq!(tensor.is_some(), data.is_some());
                match (tensor, data) {
                    (None, None) => {}
                    (Some(tensor), Some(data)) => {
                        let tensor_data = unsafe {
                            std::slice::from_raw_parts(
                                tensor.into_typed::<i32>().as_ptr(),
                                data.len(),
                            )
                        };
                        assert_eq!(&tensor_data, data);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}
