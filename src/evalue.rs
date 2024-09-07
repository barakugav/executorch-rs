//! Module for [`EValue`] and related types.
//!
//! [`EValue`] is a type-erased value that can hold different types like scalars, lists or tensors. It is used to pass
//! arguments to and return values from the runtime.

use std::fmt::Debug;
use std::pin::Pin;

use crate::error::{Error, Result};
use crate::tensor::{self, TensorAny, TensorBase};
use crate::util::{ArrayRef, Destroy, IntoRust, NonTriviallyMovable, Storable, Storage};
use crate::{et_c, et_rs_c};

/// A tag indicating the type of the value stored in an [`EValue`].
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Tag {
    /// Tag for value [`TensorAny`].
    Tensor = et_c::Tag::Tensor as u8,
    /// Tag for value `&[c_char]`.
    String = et_c::Tag::String as u8,
    /// Tag for value `f64`.
    Double = et_c::Tag::Double as u8,
    /// Tag for value `i64`.
    Int = et_c::Tag::Int as u8,
    /// Tag for value `bool`.
    Bool = et_c::Tag::Bool as u8,
    /// Tag for value `&[bool]`.
    ListBool = et_c::Tag::ListBool as u8,
    /// Tag for value `&[f64]`.
    ListDouble = et_c::Tag::ListDouble as u8,
    /// Tag for value `&[i64]`.
    ListInt = et_c::Tag::ListInt as u8,
    /// Tag for value `&[TensorAny]`.
    ListTensor = et_c::Tag::ListTensor as u8,
    /// Tag for value `Optional<TensorAny>`.
    ///
    /// Not supported at the moment.
    ListOptionalTensor = et_c::Tag::ListOptionalTensor as u8,
}
impl IntoRust for &et_c::Tag {
    type RsType = Option<Tag>;
    fn rs(self) -> Self::RsType {
        Some(match self {
            et_c::Tag::None => return None,
            et_c::Tag::Tensor => Tag::Tensor,
            et_c::Tag::String => Tag::String,
            et_c::Tag::Double => Tag::Double,
            et_c::Tag::Int => Tag::Int,
            et_c::Tag::Bool => Tag::Bool,
            et_c::Tag::ListBool => Tag::ListBool,
            et_c::Tag::ListDouble => Tag::ListDouble,
            et_c::Tag::ListInt => Tag::ListInt,
            et_c::Tag::ListTensor => Tag::ListTensor,
            et_c::Tag::ListScalar => unimplemented!("ListScalar is not supported"),
            et_c::Tag::ListOptionalTensor => Tag::ListOptionalTensor,
        })
    }
}

/// Aggregate typing system similar to IValue only slimmed down with less
/// functionality, no dependencies on atomic, and fewer supported types to better
/// suit embedded systems (ie no intrusive ptr)
pub struct EValue<'a>(NonTriviallyMovable<'a, et_c::EValue>);
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
    unsafe fn new_impl(init: impl FnOnce(*mut et_c::EValue)) -> Self {
        Self(NonTriviallyMovable::new_boxed(init))
    }

    /// Create a new [`EValue`] from a value that can be converted into an [`EValue`].
    ///
    /// The underlying Cpp object is allocated on the heap, which is preferred on systems in which allocations are
    /// available.
    /// For an identical version that allocates the object on the stack use:
    /// ```rust,ignore
    /// let storage = executorch::storage!(EValue);
    /// let evalue: EValue = storage.new(value);
    /// ```
    /// See `executorch::util::Storage` for more information.
    #[cfg(feature = "alloc")]
    pub fn new(value: impl IntoEValue<'a>) -> Self {
        value.into_evalue()
    }

    pub(crate) fn from_inner_ref(value: &'a et_c::EValue) -> Self {
        Self(NonTriviallyMovable::from_ref(value))
    }

    /// Create a new [`EValue`] by moving from an existing [`EValue`].
    ///
    /// # Safety
    ///
    /// The given value should not be used after this function is called, and its Cpp destructor should be called.
    #[cfg(feature = "alloc")]
    #[allow(dead_code)]
    pub(crate) unsafe fn move_from(value: &mut et_c::EValue) -> Self {
        Self(NonTriviallyMovable::new_boxed(|p| {
            et_rs_c::EValue_move(value, p)
        }))
    }

    pub(crate) fn as_evalue(&self) -> &et_c::EValue {
        self.0.as_ref()
    }

    /// Get a reference to the value as an `i64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not an `i64`. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_i64(&self) -> i64 {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as an `f64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not an `f64`. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_f64(&self) -> f64 {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `bool`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `bool`. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_bool(&self) -> bool {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a [`TensorAny`].
    ///
    /// # Panics
    ///
    /// Panics if the value is not a [`TensorAny`]. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_tensor(&self) -> TensorAny {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[c_char]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[c_char]`. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_chars(&self) -> &[std::ffi::c_char] {
        self.try_into().expect("Invalid type")
    }

    // /// Get a reference to the value as a `&[i64]`.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the value is not a `&[i64]`. To check the type of the value, use the [`tag`][Self::tag] method.
    // #[track_caller]
    // pub fn as_i64_arr(&self) -> &[i64] {
    //     self.try_into().expect("Invalid type")
    // }

    /// Get a reference to the value as a `&[f64]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[f64]`. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_f64_arr(&self) -> &[f64] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[bool]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[bool]`. To check the type of the value, use the [`tag`][Self::tag] method.
    #[track_caller]
    pub fn as_bool_arr(&self) -> &[bool] {
        self.try_into().expect("Invalid type")
    }

    // /// Get a reference to the value as a `&[TensorAny]`.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the value is not a `&[TensorAny]`. To check the type of the value, use the [`tag`][Self::tag] method.
    // #[track_caller]
    // pub fn as_tensor_arr(&self) -> &[TensorAny<'a>] {
    //     self.try_into().expect("Invalid type")
    // }

    /// Get the tag indicating the type of the value.
    pub fn tag(&self) -> Option<Tag> {
        self.as_evalue().tag.rs()
    }
}
impl Destroy for et_c::EValue {
    unsafe fn destroy(&mut self) {
        et_rs_c::EValue_destructor(self)
    }
}

impl Storable for EValue<'_> {
    type Storage = et_c::EValue;
}
impl Storage<EValue<'_>> {
    /// Create a new [`EValue`] from a value that can be converted into an [`EValue`] in the given storage.
    ///
    /// This function is identical to `EValue::new`, but it allows to create the evalue on the stack.
    /// See `executorch::util::Storage` for more information.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<'a>(self: Pin<&'a mut Self>, value: impl IntoEValue<'a>) -> EValue<'a> {
        value.into_evalue_in_storage(self)
    }
}

/// A type that can be converted into an [`EValue`].
pub trait IntoEValue<'a> {
    /// Convert the value into an [`EValue`], with an allocation on the heap.
    ///
    /// This is the preferred method to create an [`EValue`] when allocations are available.
    /// Use `into_evalue_in_storage` for an identical version that allow to allocate the object on the stack.
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a>;

    /// Convert the value into an [`EValue`], using the given storage.
    ///
    /// This function is identical to `into_evalue`, but it allows to create the evalue on the stack.
    /// See `executorch::util::Storage` for more information.
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(|p| et_rs_c::EValue_new_from_i64(p, self), storage)
        })
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(|p| et_rs_c::EValue_new_from_f64(p, self), storage)
        })
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(|p| et_rs_c::EValue_new_from_bool(p, self), storage)
        })
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(
                |p| et_rs_c::EValue_new_from_f64_arr(p, arr.0),
                storage,
            )
        })
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(
                |p| et_rs_c::EValue_new_from_bool_arr(p, arr.0),
                storage,
            )
        })
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(
                |p| et_rs_c::EValue_new_from_chars(p, arr.0),
                storage,
            )
        })
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
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(
                |p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor()),
                storage,
            )
        })
    }
}
impl<'a, D: tensor::Data> IntoEValue<'a> for &'a TensorBase<'_, D> {
    #[cfg(feature = "alloc")]
    fn into_evalue(self) -> EValue<'a> {
        unsafe { EValue::new_impl(|p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor())) }
    }

    fn into_evalue_in_storage(self, storage: Pin<&'a mut Storage<EValue>>) -> EValue<'a> {
        // Safety: the closure init the pointer
        EValue(unsafe {
            NonTriviallyMovable::new_in_storage(
                |p| et_rs_c::EValue_new_from_tensor(p, self.as_cpp_tensor()),
                storage,
            )
        })
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
        match value.tag() {
            Some(Tag::Int) => Ok(unsafe { *value.as_evalue().payload.copyable_union.as_int }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl TryFrom<&EValue<'_>> for f64 {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<f64> {
        match value.tag() {
            Some(Tag::Double) => Ok(unsafe { *value.as_evalue().payload.copyable_union.as_double }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl TryFrom<&EValue<'_>> for bool {
    type Error = Error;
    fn try_from(value: &EValue) -> Result<bool> {
        match value.tag() {
            Some(Tag::Bool) => Ok(unsafe { *value.as_evalue().payload.copyable_union.as_bool }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for TensorAny<'a> {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<TensorAny<'a>> {
        match value.tag() {
            Some(Tag::Tensor) => Ok(unsafe {
                let inner = &*value.as_evalue().payload.as_tensor;
                TensorAny::from_inner_ref(inner)
            }),
            _ => Err(Error::InvalidType),
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
//             _ => Err(Error::InvalidType),
//         }
//     }
// }
impl<'a> TryFrom<&'a EValue<'_>> for &'a [std::ffi::c_char] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<&'a [std::ffi::c_char]> {
        match value.tag() {
            Some(Tag::String) => Ok(unsafe {
                let arr = &*value.as_evalue().payload.copyable_union.as_string;
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
// impl<'a> TryFrom<&'a EValue<'_>> for &'a [i64] {
//     type Error = Error;
//     fn try_from(value: &'a EValue) -> Result<&'a [i64]> {
//         match value.tag() {
//             Some(Tag::ListInt) => Ok(unsafe {
//                 let arr = &*value.as_evalue().payload.copyable_union.as_int_list;
//                 BoxedEvalueList::from_inner(arr).get()
//             }),
//             _ => Err(Error::InvalidType),
//         }
//     }
// }
impl<'a> TryFrom<&'a EValue<'_>> for &'a [f64] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<&'a [f64]> {
        match value.tag() {
            Some(Tag::ListDouble) => Ok(unsafe {
                let arr = &*value.as_evalue().payload.copyable_union.as_double_list;
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&'a EValue<'_>> for &'a [bool] {
    type Error = Error;
    fn try_from(value: &'a EValue) -> Result<&'a [bool]> {
        match value.tag() {
            Some(Tag::ListBool) => Ok(unsafe {
                let arr = &*value.as_evalue().payload.copyable_union.as_bool_list;
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl Debug for EValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut st = f.debug_struct("EValue");
        st.field("tag", &self.tag());
        match self.tag() {
            Some(Tag::Int) => st.field("value", &self.as_i64()),
            Some(Tag::Double) => st.field("value", &self.as_f64()),
            Some(Tag::Bool) => st.field("value", &self.as_bool()),
            Some(Tag::Tensor) => st.field("value", &self.as_tensor()),
            Some(Tag::String) => st.field("value", &self.as_chars()),
            // Some(Tag::ListInt) => st.field("value", &self.as_i64_arr()),
            Some(Tag::ListInt) => st.field("value", &"Unsupported type"),
            Some(Tag::ListDouble) => st.field("value", &self.as_f64_arr()),
            Some(Tag::ListBool) => st.field("value", &self.as_bool_arr()),
            // Some(Tag::ListTensor) => st.field("value", &self.as_tensor_arr()),
            Some(Tag::ListTensor) => st.field("value", &"Unsupported type"),
            Some(Tag::ListOptionalTensor) => st.field("value", &"Unsupported type"),
            None => st.field("value", &"None"),
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
