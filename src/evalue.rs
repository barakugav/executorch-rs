//! Module for [`EValue`] and related types.
//!
//! [`EValue`] is a type-erased value that can hold different types like scalars, lists or tensors. It is used to pass
//! arguments to and return values from the runtime.

use std::pin::Pin;

use crate::memory::{Storable, Storage};
use crate::tensor::{self, TensorAny, TensorBase};
use crate::util::{ArrayRef, ArrayRefImpl, Destroy, NonTriviallyMovable};
use crate::{et_c, et_rs_c, Error, ErrorKind, Result};

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
/// - `ListOptionalTensor`: unsupported at the moment.
///
pub use et_c::runtime::Tag;

/// Aggregate typing system similar to IValue only slimmed down with less
/// functionality, no dependencies on atomic, and fewer supported types to better
/// suit embedded systems (ie no intrusive ptr)
pub struct EValue<'a>(NonTriviallyMovable<'a, et_c::runtime::EValue>);
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
    unsafe fn new_impl(init: impl FnOnce(*mut et_c::runtime::EValue)) -> Self {
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
        init: impl FnOnce(*mut et_c::runtime::EValue),
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
    /// let storage = pin::pin!(executorch::memory::Storage::<EValue>::default());
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

    pub(crate) fn from_inner_ref(value: &'a et_c::runtime::EValue) -> Self {
        Self(NonTriviallyMovable::from_ref(value))
    }

    /// Create a new [`EValue`] by moving from an existing [`EValue`].
    ///
    /// # Safety
    ///
    /// The given value should not be used after this function is called, and its Cpp destructor should be called.
    #[cfg(feature = "alloc")]
    #[allow(dead_code)]
    pub(crate) unsafe fn move_from(value: &mut et_c::runtime::EValue) -> Self {
        Self(unsafe { NonTriviallyMovable::new_boxed(|p| et_rs_c::EValue_move(value, p)) })
    }

    pub(crate) fn as_evalue(&self) -> &et_c::runtime::EValue {
        self.0.as_ref()
    }

    /// Create a new [`EValue`] with the no value (tag `None`).
    #[cfg(feature = "alloc")]
    pub fn none() -> Self {
        // Safety: the closure init the pointer
        let mut none = unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_i64(p, 0)) };
        unsafe { none.0.as_mut().unwrap().tag = Tag::None };
        none
    }

    /// Create a new [`EValue`] with the no value (tag `None`) in the given storage.
    pub fn none_in_storage(storage: Pin<&'a mut Storage<EValue>>) -> Self {
        // Safety: the closure init the pointer
        let mut none =
            unsafe { EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_i64(p, 0), storage) };
        unsafe { none.0.as_mut().unwrap().tag = Tag::None };
        none
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

    // /// Get a reference to the value as a `&[i64]`.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the value is of different type.
    // /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    // /// [`tag`][Self::tag] method.
    // #[track_caller]
    // pub fn as_i64_arr(&self) -> &[i64] {
    //     self.try_into().expect("Invalid type")
    // }

    /// Get a reference to the value as a `&[f64]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is of different type.
    /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    /// [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_f64_arr(&self) -> &[f64] {
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
    pub fn as_bool_arr(&self) -> &[bool] {
        self.try_into().expect("Invalid type")
    }

    // /// Get a reference to the value as a `&[TensorAny]`.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the value is of different type.
    // /// To avoid panics, use the [`try_into`][TryInto::try_into] method or check the type of the value with the
    // /// [`tag`][Self::tag] method.
    // #[track_caller]
    // pub fn as_tensor_arr(&self) -> &[TensorAny<'a>] {
    //     self.try_into().expect("Invalid type")
    // }

    /// Get the tag indicating the type of the value.
    ///
    /// Returns `None` if the inner Cpp tag is `None`.
    pub fn tag(&self) -> Tag {
        self.as_evalue().tag
    }
}
impl Destroy for et_c::runtime::EValue {
    unsafe fn destroy(&mut self) {
        unsafe { et_rs_c::EValue_destructor(self) }
    }
}

impl Storable for EValue<'_> {
    type Storage = et_c::runtime::EValue;
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
impl<'a> IntoEValue<'a> for &'a [f64] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_f64_arr(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_f64_arr(p, arr.0), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for &'a [bool] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_bool_arr(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_bool_arr(p, arr.0), storage)
        }
    }
}
impl<'a> IntoEValue<'a> for &'a [std::ffi::c_char] {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_chars(p, arr.0)) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        let arr = ArrayRef::from_slice(self);
        // Safety: the closure init the pointer
        unsafe {
            EValue::new_in_storage_impl(|p| et_rs_c::EValue_new_from_chars(p, arr.0), storage)
        }
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

// /// Create a new [`EValue`] from a list of `i64`.
// ///
// /// The functions accept two lists, one of [`EValue`] wrapping the `i64` values and one of `i64` values. See
// /// [`BoxedEvalueList`] for more information.
// ///
// /// # Arguments
// ///
// /// * `wrapped_vals` - A list of [`EValue`] wrapping the `i64` values. This is the actual values list.
// /// * `unwrapped_vals` - A mutable buffer to store the unwrapped `i64` values, used to avoid double copying. The
// /// given array can be uninitialized.
// pub fn from_i64_arr(wrapped_vals: &'a [&EValue], unwrapped_vals: &'a mut [i64]) -> Self {
//     let value = et_c::EValue_Payload_TriviallyCopyablePayload {
//         as_int_list: ManuallyDrop::new(BoxedEvalueList::new(wrapped_vals, unwrapped_vals).0),
//     };
//     unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListInt) }
// }

// /// Create a new [`EValue`] from a list of [`Tensor`].
// ///
// /// The functions accept two lists, one of [`EValue`] wrapping the [`Tensor`] values and one of [`Tensor`] values. See
// /// [`BoxedEvalueList`] for more information.
// ///
// /// # Arguments
// ///
// /// * `wrapped_vals` - A list of [`EValue`] wrapping the [`Tensor`] values. This is the actual values list.
// /// * `unwrapped_vals` - A mutable buffer to store the unwrapped [`Tensor`] values, used to avoid double copying. The
// /// given array can be uninitialized.
// pub fn from_tensor_arr(
//     wrapped_vals: &'a [&EValue],
//     unwrapped_vals: &'a mut [Tensor<'a>],
// ) -> Self {
//     let list = BoxedEvalueList::new(wrapped_vals, unwrapped_vals).0;
//     // Safety: Tensor and et_c::Tensor have the same memory layout
//     let list = unsafe {
//         std::mem::transmute::<
//             et_c::BoxedEvalueList<Tensor<'a>>,
//             et_c::BoxedEvalueList<et_c::Tensor>,
//         >(list)
//     };
//     let value = et_c::EValue_Payload_TriviallyCopyablePayload {
//         as_tensor_list: ManuallyDrop::new(list),
//     };
//     unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListTensor) }
// }
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
    fn try_from(value: &EValue) -> Result<i64> {
        if value.tag() == Tag::Int {
            Ok(unsafe { et_rs_c::EValue_as_i64(value.as_evalue()) })
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
        }
    }
}
impl TryFrom<&EValue<'_>> for f64 {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<f64> {
        if value.tag() == Tag::Double {
            Ok(unsafe { et_rs_c::EValue_as_f64(value.as_evalue()) })
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
        }
    }
}
impl TryFrom<&EValue<'_>> for bool {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<bool> {
        if value.tag() == Tag::Bool {
            Ok(unsafe { et_rs_c::EValue_as_bool(value.as_evalue()) })
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for TensorAny<'a> {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<TensorAny<'a>> {
        if value.tag() == Tag::Tensor {
            let inner = unsafe { &*value.as_evalue().payload.as_tensor };
            Ok(TensorAny::from_inner_ref(inner))
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
        }
    }
}
// impl<'a> TryFrom<EValue<'a>> for Tensor<'a> {
//     type Error = Error;
//     fn try_from(mut value: EValue<'a>) -> Result<Tensor<'a>> {
//         if value.tag() == Tag::Tensor {
//             Ok(unsafe {
//                 value.0.tag = et_c::Tag::None;
//                 let inner = ManuallyDrop::take(&mut value.0.payload.as_tensor);
//                 Tensor::from_inner(inner)
//             })
//         } else {
//             Err(Error::simple(ErrorKind::InvalidType))
//         }
//     }
// }
impl<'a> TryFrom<&'a EValue<'_>> for &'a [std::ffi::c_char] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<&'a [std::ffi::c_char]> {
        if value.tag() == Tag::String {
            Ok(unsafe { et_rs_c::EValue_as_string(value.as_evalue()).as_slice() })
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
        }
    }
}
// impl<'a> TryFrom<&'a EValue<'_>> for &'a [i64] {
//     type Error = Error;
//     fn try_from(value: &'a EValue) -> Result<&'a [i64]> {
//         if value.tag() == Tag::ListInt {
//             Ok(unsafe {
//                 let arr = &*value.as_evalue().payload.copyable_union.as_int_list;
//                 BoxedEvalueList::from_inner(arr).get()
//             })
//         } else {
//             Err(Error::simple(ErrorKind::InvalidType))
//         }
//     }
// }
impl<'a> TryFrom<&'a EValue<'_>> for &'a [f64] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<&'a [f64]> {
        if value.tag() == Tag::ListDouble {
            Ok(unsafe { et_rs_c::EValue_as_f64_list(value.as_evalue()).as_slice() })
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [bool] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<&'a [bool]> {
        if value.tag() == Tag::ListBool {
            Ok(unsafe { et_rs_c::EValue_as_bool_list(value.as_evalue()).as_slice() })
        } else {
            Err(Error::simple(ErrorKind::InvalidType))
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
            // Tag::ListInt => st.field("value", &self.as_i64_arr()),
            Tag::ListInt => st.field("value", &"Unsupported type"),
            Tag::ListDouble => st.field("value", &self.as_f64_arr()),
            Tag::ListBool => st.field("value", &self.as_bool_arr()),
            // Tag::ListTensor => st.field("value", &self.as_tensor_arr()),
            Tag::ListTensor => st.field("value", &"Unsupported type"),
            Tag::ListOptionalTensor => st.field("value", &"Unsupported type"),
            Tag::ListScalar => st.field("value", &"Unsupported type"),
            Tag::None => st.field("value", &"None"),
        };
        st.finish()
    }
}

// /// Helper class used to correlate EValues in the executor table, with the
// /// unwrapped list of the proper type. Because values in the runtime's values
// /// table can change during execution, we cannot statically allocate list of
// /// objects at deserialization. Imagine the serialized list says index 0 in the
// /// value table is element 2 in the list, but during execution the value in
// /// element 2 changes (in the case of tensor this means the &TensorImpl stored in
// /// the tensor changes). To solve this instead they must be created dynamically
// /// whenever they are used.
// pub struct BoxedEvalueList<'a, T: BoxedEvalue>(et_c::BoxedEvalueList<T>, PhantomData<&'a ()>);
// impl<'a, T: BoxedEvalue> BoxedEvalueList<'a, T> {
//     pub(crate) unsafe fn from_inner(inner: &et_c::BoxedEvalueList<T>) -> &Self {
//         // Safety: BoxedEvalueList and et_c::BoxedEvalueList have the same memory layout
//         std::mem::transmute::<&et_c::BoxedEvalueList<T>, &BoxedEvalueList<T>>(inner)
//     }

//     /// Wrapped_vals is a list of pointers into the values table of the runtime
//     /// whose destinations correlate with the elements of the list, unwrapped_vals
//     /// is a container of the same size whose serves as memory to construct the
//     /// unwrapped vals.
//     pub fn new(wrapped_vals: &'a [&EValue], unwrapped_vals: &'a mut [T]) -> Self {
//         assert_eq!(
//             wrapped_vals.len(),
//             unwrapped_vals.len(),
//             "Length mismatch between wrapped and unwrapped values"
//         );
//         assert!(
//             wrapped_vals.iter().all(|val| val.tag() == Some(T::TAG)),
//             "wrapped_vals contains type different from T"
//         );

//         let wrapped_vals = ArrayRef::from_slice(wrapped_vals).0;
//         // Safety: EValue and et_c::EValue have the same memory layout
//         let wrapped_vals = unsafe {
//             std::mem::transmute::<et_c::ArrayRef<&EValue>, et_c::ArrayRef<&et_c::EValue>>(
//                 wrapped_vals,
//             )
//         };
//         // Safety: &et_c::EValue and *mut et_c::EValue have the same memory layout, and ArrayRef is an immutable type
//         // so it's safe to transmute its inner values to a mutable type as they will not be mutated
//         let wrapped_vals = unsafe {
//             std::mem::transmute::<et_c::ArrayRef<&et_c::EValue>, et_c::ArrayRef<*mut et_c::EValue>>(
//                 wrapped_vals,
//             )
//         };

//         let list = et_c::BoxedEvalueList {
//             wrapped_vals_: wrapped_vals,
//             unwrapped_vals_: unwrapped_vals.as_mut_ptr(),
//             _phantom_0: PhantomData,
//         };
//         Self(list, PhantomData)
//     }

//     /// Constructs and returns the list of T specified by the EValue pointers
//     pub fn get(&self) -> &'a [T] {
//         let evalues = unsafe { ArrayRef::from_inner(&self.0.wrapped_vals_) }.as_slice();
//         // Safety: EValue and et_c::EValue have the same memory layout
//         let evalues = unsafe { std::mem::transmute::<&[*mut et_c::EValue], &[&EValue]>(evalues) };
//         assert!(
//             evalues.iter().all(|val| val.tag() == Some(T::TAG)),
//             "EValues have different tags"
//         );

//         let unwrapped_list = match T::TAG {
//             Tag::Int => {
//                 // Safety: T is i64
//                 let list = unsafe {
//                     std::mem::transmute::<&BoxedEvalueList<'a, T>, &BoxedEvalueList<'a, i64>>(self)
//                 };
//                 let unwrapped_list = unsafe { et_rs_c::BoxedEvalueList_i64_get(&list.0) };
//                 // Safety: i64 is T
//                 unsafe {
//                     std::mem::transmute::<et_c::ArrayRef<i64>, et_c::ArrayRef<T>>(unwrapped_list)
//                 }
//             }
//             Tag::Tensor => {
//                 // Safety: T is Tensor
//                 let list = unsafe {
//                     std::mem::transmute::<&BoxedEvalueList<'a, T>, &BoxedEvalueList<'a, Tensor<'_>>>(
//                         self,
//                     )
//                 };
//                 // Safety: Tensor and et_c::Tensor have the same memory layout
//                 let list = unsafe {
//                     std::mem::transmute::<
//                         &et_c::BoxedEvalueList<Tensor<'_>>,
//                         &et_c::BoxedEvalueList<et_c::Tensor>,
//                     >(&list.0)
//                 };
//                 let unwrapped_list = unsafe { et_rs_c::BoxedEvalueList_Tensor_get(list) };
//                 // Safety: et_c::Tensor and Tensor have the same memory layout
//                 let unwrapped_list = unsafe {
//                     std::mem::transmute::<et_c::ArrayRef<et_c::Tensor>, et_c::ArrayRef<Tensor<'_>>>(
//                         unwrapped_list,
//                     )
//                 };
//                 // Safety: Tensor is T
//                 unsafe {
//                     std::mem::transmute::<et_c::ArrayRef<Tensor<'a>>, et_c::ArrayRef<T>>(
//                         unwrapped_list,
//                     )
//                 }
//             }
//             unsupported_type => panic!("Unsupported type: {:?}", unsupported_type),
//         };
//         unsafe { ArrayRef::from_inner(&unwrapped_list).as_slice() }
//     }
// }
// impl<T: BoxedEvalue + Debug> Debug for BoxedEvalueList<'_, T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         self.get().fmt(f)
//     }
// }

// /// A trait for types that can be used within a [`BoxedEvalueList`].
// pub trait BoxedEvalue {
//     /// The [`Tag`] variant corresponding to boxed type.
//     const TAG: Tag;
//     private_decl! {}
// }
// impl BoxedEvalue for i64 {
//     const TAG: Tag = Tag::Int;
//     private_impl! {}
// }
// impl BoxedEvalue for Tensor<'_> {
//     const TAG: Tag = Tag::Tensor;
//     private_impl! {}
// }
