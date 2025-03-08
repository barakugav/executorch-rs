//! Module for [`EValue`] and related types.
//!
//! [`EValue`] is a type-erased value that can hold different types like scalars, lists or tensors. It is used to pass
//! arguments to and return values from the runtime.

use std::ffi::CStr;
use std::pin::Pin;

use crate::memory::{MemoryAllocator, Storable, Storage};
use crate::tensor::{self, TensorAny, TensorBase};
use crate::util::{ArrayRef, Destroy, IntoCpp, IntoRust, NonTriviallyMovable, __ArrayRefImpl};
use crate::{CError, Error, Result};
use executorch_sys as et_c;

/// A tag indicating the type of the value stored in an [`EValue`].
#[repr(u32)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Tag {
    /// Tag for an empty EValue.
    None = 0,
    /// Tag for value [`TensorAny`].
    Tensor = 1,
    /// Tag for value `&[c_char]`.
    String = 2,
    /// Tag for value `f64`.
    Double = 3,
    /// Tag for value `i64`.
    Int = 4,
    /// Tag for value `bool`.
    Bool = 5,
    /// Tag for value `&[bool]`.
    ListBool = 6,
    /// Tag for value `&[f64]`.
    ListDouble = 7,
    /// Tag for value `&[i64]`.
    ListInt = 8,
    /// Tag for value `&[TensorAny]`.
    ListTensor = 9,
    /// unsupported at the moment.
    ListScalar = 10,
    /// Tag for value `&[Option<TensorAny>]`.
    ListOptionalTensor = 11,
}

impl IntoRust for et_c::Tag {
    type RsType = Tag;
    fn rs(self) -> Self::RsType {
        match self {
            et_c::Tag::Tag_None => Tag::None,
            et_c::Tag::Tag_Tensor => Tag::Tensor,
            et_c::Tag::Tag_String => Tag::String,
            et_c::Tag::Tag_Double => Tag::Double,
            et_c::Tag::Tag_Int => Tag::Int,
            et_c::Tag::Tag_Bool => Tag::Bool,
            et_c::Tag::Tag_ListBool => Tag::ListBool,
            et_c::Tag::Tag_ListDouble => Tag::ListDouble,
            et_c::Tag::Tag_ListInt => Tag::ListInt,
            et_c::Tag::Tag_ListTensor => Tag::ListTensor,
            et_c::Tag::Tag_ListScalar => Tag::ListScalar,
            et_c::Tag::Tag_ListOptionalTensor => Tag::ListOptionalTensor,
        }
    }
}

/// Aggregate typing system similar to IValue only slimmed down with less
/// functionality, no dependencies on atomic, and fewer supported types to better
/// suit embedded systems (ie no intrusive ptr)
pub struct EValue<'a>(NonTriviallyMovable<'a, et_c::EValueStorage>);
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
    unsafe fn new_impl(init: impl FnOnce(et_c::EValueRefMut)) -> Self {
        let init = |ptr: *mut et_c::EValueStorage| init(et_c::EValueRefMut { ptr: ptr as *mut _ });
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
        init: impl FnOnce(et_c::EValueRefMut),
        storage: Pin<&'a mut Storage<EValue>>,
    ) -> Self {
        let init = |ptr: *mut et_c::EValueStorage| init(et_c::EValueRefMut { ptr: ptr as *mut _ });
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

    /// Create a new [`EValue`] from a value that can be converted into one in the given memory allocator.
    ///
    /// This function is identical to [`EValue::new_in_storage`][EValue::new_in_storage], but it allocates the storage
    /// using the given memory allocator.
    ///
    /// # Panics
    ///
    /// If the allocation fails.
    pub fn new_in_allocator(
        value: impl IntoEValue<'a>,
        allocator: &'a MemoryAllocator<'a>,
    ) -> Self {
        let storage = allocator
            .allocate_pinned()
            .ok_or(Error::CError(CError::MemoryAllocationFailed))
            .unwrap();
        Self::new_in_storage(value, storage)
    }

    pub(crate) unsafe fn from_inner_ref(value: et_c::EValueRef) -> Self {
        let value = value.ptr as *const et_c::EValueStorage;
        assert!(!value.is_null());
        let value = unsafe { &*value };
        Self(NonTriviallyMovable::from_ref(value))
    }

    /// Create a new [`EValue`] by moving from an existing [`EValue`].
    ///
    /// # Safety
    ///
    /// The given value should not be used after this function is called, and its Cpp destructor should be called.
    #[cfg(feature = "alloc")]
    #[allow(dead_code)]
    pub(crate) unsafe fn move_from(value: et_c::EValueRefMut) -> Self {
        Self(unsafe {
            NonTriviallyMovable::new_boxed(|p: *mut et_c::EValueStorage| {
                et_c::executorch_EValue_move(value, et_c::EValueRefMut { ptr: p as *mut _ })
            })
        })
    }

    /// Create a new [`EValue`] with the no value (tag `None`).
    #[cfg(feature = "alloc")]
    pub fn none() -> Self {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_none(p)) }
    }

    /// Create a new [`EValue`] with the no value (tag `None`) in the given storage.
    pub fn none_in_storage(storage: Pin<&'a mut Storage<EValue>>) -> Self {
        // Safety: the closure init the pointer
        unsafe { EValue::new_in_storage_impl(|p| et_c::executorch_EValue_new_none(p), storage) }
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
        self.try_into().unwrap()
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
        self.try_into().unwrap()
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
        self.try_into().unwrap()
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
        self.try_into().unwrap()
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
        self.try_into().unwrap()
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
        self.try_into().unwrap()
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
        self.try_into().unwrap()
    }

    /// Get a reference to the value as a `CStr`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    pub fn as_cstr(&self) -> &CStr {
        self.try_into().unwrap()
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
        self.try_into().unwrap()
    }

    /// Get a reference to the value as a [`TensorList`].
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    pub fn as_tensor_list(&self) -> TensorList {
        self.try_into().unwrap()
    }

    /// Get a reference to the value as a [`OptionalTensorList`].
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    pub fn as_optional_tensor_list(&self) -> OptionalTensorList {
        self.try_into().unwrap()
    }

    /// Get the tag indicating the type of the value.
    ///
    /// Returns `None` if the inner Cpp tag is `None`.
    pub fn tag(&self) -> Tag {
        unsafe { et_c::executorch_EValue_tag(self.cpp()) }.rs()
    }
}
impl Destroy for et_c::EValueStorage {
    unsafe fn destroy(&mut self) {
        unsafe {
            et_c::executorch_EValue_destructor(et_c::EValueRefMut {
                ptr: self as *mut Self as *mut _,
            })
        }
    }
}
impl IntoCpp for &EValue<'_> {
    type CppType = et_c::EValueRef;
    fn cpp(self) -> Self::CppType {
        et_c::EValueRef {
            ptr: self.0.as_ref() as *const et_c::EValueStorage as *const _,
        }
    }
}

impl Storable for EValue<'_> {
    type __Storage = et_c::EValueStorage;
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
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_i64(p, self)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_c::executorch_EValue_new_from_i64(p, self), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for BoxedEvalueList<'a, i64> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_i64_list(p, self.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_i64_list(p, self.0),
                storage,
            )
        }
    }
}
impl<'a> IntoEValue<'a> for f64 {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_f64(p, self)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_c::executorch_EValue_new_from_f64(p, self), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for &'a [f64] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_f64_list(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_f64_list(p, arr.0),
                storage,
            )
        }
    }
}
impl<'a> IntoEValue<'a> for bool {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_bool(p, self)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_c::executorch_EValue_new_from_bool(p, self), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for &'a [bool] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_bool_list(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_bool_list(p, arr.0),
                storage,
            )
        }
    }
}
impl<'a> IntoEValue<'a> for &'a [std::ffi::c_char] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_string(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_string(p, arr.0),
                storage,
            )
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
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_tensor(p, self.as_cpp())) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_tensor(p, self.as_cpp()),
                storage,
            )
        }
    }
}
impl<'a, D: tensor::Data> IntoEValue<'a> for &'a TensorBase<'_, D> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_tensor(p, self.as_cpp())) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_tensor(p, self.as_cpp()),
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
        unsafe { EValue::new_impl(|p| et_c::executorch_EValue_new_from_tensor_list(p, self.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_tensor_list(p, self.0),
                storage,
            )
        }
    }
}
impl<'a> IntoEValue<'a> for BoxedEvalueList<'a, Option<TensorAny<'a>>> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_impl(|p| et_c::executorch_EValue_new_from_optional_tensor_list(p, self.0))
        }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(
                |p| et_c::executorch_EValue_new_from_optional_tensor_list(p, self.0),
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
            Ok(unsafe { et_c::executorch_EValue_as_i64(value.cpp()) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [i64] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListInt {
            Ok(unsafe { et_c::executorch_EValue_as_i64_list(value.cpp()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl TryFrom<&EValue<'_>> for f64 {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<Self> {
        if value.tag() == Tag::Double {
            Ok(unsafe { et_c::executorch_EValue_as_f64(value.cpp()) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [f64] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListDouble {
            Ok(unsafe { et_c::executorch_EValue_as_f64_list(value.cpp()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl TryFrom<&EValue<'_>> for bool {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<Self> {
        if value.tag() == Tag::Bool {
            Ok(unsafe { et_c::executorch_EValue_as_bool(value.cpp()) })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [bool] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::ListBool {
            Ok(unsafe { et_c::executorch_EValue_as_bool_list(value.cpp()).as_slice() })
        } else {
            Err(Error::CError(CError::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [std::ffi::c_char] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<Self> {
        if value.tag() == Tag::String {
            Ok(unsafe { et_c::executorch_EValue_as_string(value.cpp()).as_slice() })
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
            let tensor = unsafe { et_c::executorch_EValue_as_tensor(value.cpp()) };
            Ok(unsafe { TensorAny::from_inner_ref(tensor) })
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
            let list = unsafe { et_c::executorch_EValue_as_tensor_list(value.cpp()) };
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
            let list = unsafe { et_c::executorch_EValue_as_optional_tensor_list(value.cpp()) };
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
pub struct TensorList<'a>(&'a [et_c::TensorStorage]);
impl TensorList<'_> {
    /// Safety: the array must be valid for the lifetime of the returned list.
    unsafe fn from_array_ref(array: et_c::ArrayRefTensor) -> Self {
        let data = array.data.ptr as *const et_c::TensorStorage;
        Self(unsafe { std::slice::from_raw_parts(data, array.len) })
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
        self.0.get(index).map(|t| unsafe {
            TensorAny::from_inner_ref(et_c::TensorRef {
                ptr: t as *const et_c::TensorStorage as *const _,
            })
        })
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
pub struct OptionalTensorList<'a>(&'a [et_c::OptionalTensorStorage]);
impl OptionalTensorList<'_> {
    /// Safety: the array must be valid for the lifetime of the returned list.
    unsafe fn from_array_ref(array: et_c::ArrayRefOptionalTensor) -> Self {
        let data = array.data.ptr as *const et_c::OptionalTensorStorage;
        Self(unsafe { std::slice::from_raw_parts(data, array.len) })
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
        self.0.get(index).map(|tensor| {
            let tensor = et_c::OptionalTensorRef {
                ptr: tensor as *const et_c::OptionalTensorStorage as *const _,
            };
            let tensor = unsafe { et_c::executorch_OptionalTensor_get(tensor) };
            if tensor.ptr.is_null() {
                return None;
            }
            Some(unsafe { TensorAny::from_inner_ref(tensor) })
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

mod boxed_list;
pub use boxed_list::*;

#[cfg(test)]
mod tests {
    use crate::memory::BufferMemoryAllocator;
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

            let evalue: EValue = 17.into();
            assert_eq!(evalue.tag(), Tag::Int);
            assert_eq!(evalue.as_i64(), 17);

            test_try_from_evalue(Tag::Int);
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

            test_try_from_evalue(Tag::ListInt);
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

            test_try_from_evalue(Tag::Double);
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

            test_try_from_evalue(Tag::ListDouble);
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

            test_try_from_evalue(Tag::Bool);
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

            test_try_from_evalue(Tag::ListBool);
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
        let string = cstr::cstr!("hello world!");
        let chars = crate::util::cstr2chars(string);

        #[cfg(feature = "alloc")]
        {
            let evalue = EValue::new(string);
            assert_eq!(evalue.tag(), Tag::String);
            assert_eq!(evalue.as_cstr(), string);
            assert_eq!(evalue.as_chars(), chars);

            test_try_from_evalue(Tag::String);
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
            let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();
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

            test_try_from_evalue(Tag::Tensor);
        }
        #[cfg(feature = "tensor-ptr")]
        {
            let tensor = TensorPtr::from_slice(&data);
            let evalue = EValue::new(&tensor);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);

            let tensor = TensorPtr::from_slice(&data);
            let evalue_storage = storage!(EValue);
            let evalue = EValue::new_in_storage(&tensor, evalue_storage);
            assert_eq!(evalue.tag(), Tag::Tensor);
            let tensor = evalue.as_tensor().into_typed::<i32>();
            let tensor_data = unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
            assert_eq!(tensor_data, data);
        }
        {
            let sizes = [data.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();
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
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides).unwrap();
            let tensor1 = Tensor::new(&tensor_impl);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides).unwrap();
            let tensor2 = Tensor::new(&tensor_impl);

            let sizes = [data3.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data3, &dim_order, &strides).unwrap();
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
            assert!(!tensor_list.is_empty());

            for (i, data) in [data1.as_slice(), &data2, &data3].iter().enumerate() {
                let tensor = tensor_list.get(i).unwrap().into_typed::<i32>();
                let tensor_data =
                    unsafe { std::slice::from_raw_parts(tensor.as_ptr(), data.len()) };
                assert_eq!(&tensor_data, data);
            }

            test_try_from_evalue(Tag::ListTensor);
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
            assert!(!tensor_list.is_empty());

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
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides).unwrap();
            let tensor_storage = storage!(Tensor<i32>);
            let tensor1 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides).unwrap();
            let tensor_storage = storage!(Tensor<i32>);
            let tensor2 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let sizes = [data3.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data3, &dim_order, &strides).unwrap();
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
            assert!(!tensor_list.is_empty());

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
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides).unwrap();
            let tensor1 = Tensor::new(&tensor_impl);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides).unwrap();
            let tensor2 = Tensor::new(&tensor_impl);

            // let tensor3 = None;

            let sizes = [data4.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data4, &dim_order, &strides).unwrap();
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
            assert!(!tensor_list.is_empty());

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

            test_try_from_evalue(Tag::ListOptionalTensor);
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
            assert!(!tensor_list.is_empty());

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
            let tensor_impl = TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides).unwrap();
            let tensor_storage = storage!(Tensor<i32>);
            let tensor1 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            let sizes = [data2.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides).unwrap();
            let tensor_storage = storage!(Tensor<i32>);
            let tensor2 = Tensor::new_in_storage(&tensor_impl, tensor_storage);

            // let tensor3 = None;

            let sizes = [data4.len() as SizesType];
            let dim_order = [0];
            let strides = [1];
            let tensor_impl = TensorImpl::from_slice(&sizes, &data4, &dim_order, &strides).unwrap();
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
            assert!(!tensor_list.is_empty());

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

    #[cfg(feature = "alloc")]
    fn test_try_from_evalue(tag: Tag) {
        let tags = [
            Tag::None,
            Tag::Tensor,
            Tag::String,
            Tag::Double,
            Tag::Int,
            Tag::Bool,
            Tag::ListBool,
            Tag::ListDouble,
            Tag::ListInt,
            Tag::ListTensor,
            Tag::ListOptionalTensor,
        ];

        for actual_tag in tags {
            let check_evalue = |evalue: EValue<'_>| {
                let same_tag = actual_tag == tag;
                match tag {
                    Tag::None => unimplemented!(),
                    Tag::Tensor => assert_eq!(
                        same_tag,
                        <TensorAny as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::String => {
                        assert_eq!(
                            same_tag,
                            <&[std::ffi::c_char] as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                        );
                        assert_eq!(
                            same_tag,
                            <&CStr as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                        );
                    }
                    Tag::Double => assert_eq!(
                        same_tag,
                        <f64 as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::Int => assert_eq!(
                        same_tag,
                        <i64 as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::Bool => assert_eq!(
                        same_tag,
                        <bool as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::ListBool => assert_eq!(
                        same_tag,
                        <&[bool] as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::ListDouble => assert_eq!(
                        same_tag,
                        <&[f64] as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::ListInt => assert_eq!(
                        same_tag,
                        <&[i64] as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::ListTensor => assert_eq!(
                        same_tag,
                        <TensorList as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::ListOptionalTensor => assert_eq!(
                        same_tag,
                        <OptionalTensorList as TryFrom<&EValue>>::try_from(&evalue).is_ok()
                    ),
                    Tag::ListScalar => unimplemented!(),
                }
            };
            match actual_tag {
                Tag::None => check_evalue(EValue::none()),
                Tag::Tensor => {
                    let data: [i32; 3] = [42, 17, 6];
                    let sizes = [data.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data, &dim_order, &strides).unwrap();
                    check_evalue(EValue::new(Tensor::new(&tensor_impl)));
                }
                Tag::String => check_evalue(EValue::new(cstr::cstr!("hello world!"))),
                Tag::Double => check_evalue(EValue::new(42.6)),
                Tag::Int => check_evalue(EValue::new(17)),
                Tag::Bool => check_evalue(EValue::new(true)),
                Tag::ListBool => {
                    let bool_list = [true, false, false, true, true];
                    check_evalue(EValue::new(bool_list.as_slice()))
                }
                Tag::ListDouble => {
                    let f64_list = [42.2_f64, 17.6, 55.9];
                    check_evalue(EValue::new(f64_list.as_slice()))
                }
                Tag::ListInt => {
                    let values = [42_i64, 17, 99].map(EValue::new);
                    let wrapped_vals = EValuePtrList::new([&values[0], &values[1], &values[2]]);
                    let mut unwrapped_vals = storage!(i64, (3));
                    let list =
                        BoxedEvalueList::new(&wrapped_vals, unwrapped_vals.as_mut()).unwrap();
                    check_evalue(EValue::new(list));
                }
                Tag::ListTensor => {
                    let data1: [i32; 3] = [42, 17, 6];
                    let sizes = [data1.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides).unwrap();
                    let tensor1 = Tensor::new(&tensor_impl);

                    let data2: [i32; 2] = [55, 8];
                    let sizes = [data2.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides).unwrap();
                    let tensor2 = Tensor::new(&tensor_impl);

                    let data3: [i32; 2] = [106, 144];
                    let sizes = [data3.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data3, &dim_order, &strides).unwrap();
                    let tensor3 = Tensor::new(&tensor_impl);

                    let evalue1 = EValue::new(tensor1);
                    let evalue2 = EValue::new(tensor2);
                    let evalue3 = EValue::new(tensor3);
                    let wrapped_vals = EValuePtrList::new([&evalue1, &evalue2, &evalue3]);
                    let unwrapped_vals = storage!(TensorAny, [3]);
                    let list = BoxedEvalueList::new(&wrapped_vals, unwrapped_vals).unwrap();
                    check_evalue(EValue::new(list));
                }
                Tag::ListOptionalTensor => {
                    let data1: [i32; 3] = [42, 17, 6];

                    let sizes = [data1.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data1, &dim_order, &strides).unwrap();
                    let tensor1 = Tensor::new(&tensor_impl);

                    let data2: [i32; 2] = [55, 8];
                    let sizes = [data2.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data2, &dim_order, &strides).unwrap();
                    let tensor2 = Tensor::new(&tensor_impl);

                    // let tensor3 = None;

                    let data4: [i32; 2] = [106, 144];
                    let sizes = [data4.len() as SizesType];
                    let dim_order = [0];
                    let strides = [1];
                    let tensor_impl =
                        TensorImpl::from_slice(&sizes, &data4, &dim_order, &strides).unwrap();
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
                    check_evalue(EValue::new(list));
                }
                Tag::ListScalar => unimplemented!(),
            }
        }
    }

    #[test]
    fn new_in_allocator() {
        let mut allocator_buf = [0_u8; 512];
        let allocator = BufferMemoryAllocator::new(&mut allocator_buf);

        let evalue_int = EValue::new_in_allocator(17, &allocator);
        let evalue_float = EValue::new_in_allocator(42.6, &allocator);
        assert_eq!(evalue_int.as_i64(), 17);
        assert_eq!(evalue_float.as_f64(), 42.6);
    }
}
