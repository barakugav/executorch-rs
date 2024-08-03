use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use crate::util::{ArrayRef, IntoRust};
use crate::{et_c, et_rs_c, tensor::Tensor, Error, Result};

/// A tag indicating the type of the value stored in an `EValue`.
#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Tag {
    /// Tag for value `Tensor`.
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
    /// Tag for value `&[Tensor]`.
    ListTensor = et_c::Tag::ListTensor as u8,
    /// Tag for value `Optional<Tensor>`.
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
pub struct EValue<'a>(pub(crate) et_c::EValue, PhantomData<&'a ()>);
impl<'a> EValue<'a> {
    unsafe fn new(value: et_c::EValue) -> Self {
        Self(value, PhantomData)
    }

    unsafe fn new_trivially_copyable(
        value: et_c::EValue_Payload_TriviallyCopyablePayload,
        tag: et_c::Tag,
    ) -> Self {
        unsafe {
            EValue::new(et_c::EValue {
                payload: et_c::EValue_Payload {
                    copyable_union: ManuallyDrop::new(value),
                },
                tag,
            })
        }
    }

    /// Create a new `EValue` from an `i64`.
    pub fn from_i64(val: i64) -> EValue<'static> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_int: ManuallyDrop::new(val),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::Int) }
    }

    /// Create a new `EValue` from an `f64`.
    pub fn from_f64(val: f64) -> EValue<'static> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_double: ManuallyDrop::new(val),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::Double) }
    }

    /// Create a new `EValue` from a `bool`.
    pub fn from_bool(val: bool) -> EValue<'static> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_bool: ManuallyDrop::new(val),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::Bool) }
    }

    /// Create a new `EValue` from a `Tensor`.
    pub fn from_tensor(tensor: Tensor<'a>) -> EValue<'a> {
        unsafe {
            EValue::new(et_c::EValue {
                payload: et_c::EValue_Payload {
                    as_tensor: ManuallyDrop::new(tensor.0),
                },
                tag: et_c::Tag::Tensor,
            })
        }
    }

    /// Create a new `EValue` from a `&[c_char]`, aka string.
    pub fn from_chars(chars: &'a [std::os::raw::c_char]) -> EValue<'a> {
        let chars = ArrayRef::from_slice(chars);
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_string: ManuallyDrop::new(chars.0),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::String) }
    }

    /// Create a new `EValue` from a list of `i64`.
    ///
    /// The functions accept two lists, one of `EValue` wrapping the `i64` values and one of `i64` values. See
    /// `BoxedEvalueList` for more information.
    ///
    /// # Arguments
    ///
    /// * `wrapped_vals` - A list of `EValue` wrapping the `i64` values. This is the actual values list.
    /// * `unwrapped_vals` - A mutable buffer to store the unwrapped `i64` values, used to avoid double copying. The
    /// given array can be uninitialized.
    pub fn from_i64_arr(wrapped_vals: &'a [&EValue], unwrapped_vals: &'a mut [i64]) -> Self {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_int_list: ManuallyDrop::new(BoxedEvalueList::new(wrapped_vals, unwrapped_vals).0),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListInt) }
    }

    /// Create a new `EValue` from a list of `f64`.
    pub fn from_f64_arr(arr: &'a [f64]) -> EValue<'a> {
        let arr = ArrayRef::from_slice(arr);
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_double_list: ManuallyDrop::new(arr.0),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListDouble) }
    }

    /// Create a new `EValue` from a list of `bool`.
    pub fn from_bool_arr(arr: &'a [bool]) -> EValue<'a> {
        let arr = ArrayRef::from_slice(arr);
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_bool_list: ManuallyDrop::new(arr.0),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListBool) }
    }

    /// Create a new `EValue` from a list of `Tensor`.
    ///
    /// The functions accept two lists, one of `EValue` wrapping the `Tensor` values and one of `Tensor` values. See
    /// `BoxedEvalueList` for more information.
    ///
    /// # Arguments
    ///
    /// * `wrapped_vals` - A list of `EValue` wrapping the `Tensor` values. This is the actual values list.
    /// * `unwrapped_vals` - A mutable buffer to store the unwrapped `Tensor` values, used to avoid double copying. The
    /// given array can be uninitialized.
    pub fn from_tensor_arr(
        wrapped_vals: &'a [&EValue],
        unwrapped_vals: &'a mut [Tensor<'a>],
    ) -> Self {
        let list = BoxedEvalueList::new(wrapped_vals, unwrapped_vals).0;
        // SAFETY: Tensor and et_c::Tensor have the same memory layout
        let list = unsafe {
            std::mem::transmute::<
                et_c::BoxedEvalueList<Tensor<'a>>,
                et_c::BoxedEvalueList<et_c::Tensor>,
            >(list)
        };
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_tensor_list: ManuallyDrop::new(list),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListTensor) }
    }

    /// Get a reference to the value as an `i64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not an `i64`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_i64(&self) -> i64 {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as an `f64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not an `f64`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_f64(&self) -> f64 {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `bool`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `bool`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_bool(&self) -> bool {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `Tensor`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `Tensor`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_tensor(&self) -> &Tensor<'a> {
        self.try_into().expect("Invalid type")
    }

    /// Convert the value into a `Tensor`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `Tensor`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn into_tensor(self) -> Tensor<'a> {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[c_char]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[c_char]`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_chars(&self) -> &'a [std::os::raw::c_char] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[i64]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[i64]`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_i64_arr(&self) -> &'a [i64] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[f64]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[f64]`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_f64_arr(&self) -> &'a [f64] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[bool]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[bool]`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_bool_arr(&self) -> &'a [bool] {
        self.try_into().expect("Invalid type")
    }

    /// Get a reference to the value as a `&[Tensor]`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not a `&[Tensor]`. To check the type of the value, use the `tag` method.
    #[track_caller]
    pub fn as_tensor_arr(&self) -> &[Tensor<'a>] {
        self.try_into().expect("Invalid type")
    }

    /// Get the tag indicating the type of the value.
    pub fn tag(&self) -> Option<Tag> {
        self.0.tag.rs()
    }
}
impl Drop for EValue<'_> {
    fn drop(&mut self) {
        unsafe { et_rs_c::EValue_destructor(&mut self.0) }
    }
}

impl From<i64> for EValue<'static> {
    fn from(val: i64) -> Self {
        Self::from_i64(val)
    }
}
impl From<f64> for EValue<'static> {
    fn from(val: f64) -> Self {
        Self::from_f64(val)
    }
}
impl From<bool> for EValue<'static> {
    fn from(val: bool) -> Self {
        Self::from_bool(val)
    }
}
impl<'a> From<&'a [std::os::raw::c_char]> for EValue<'a> {
    fn from(chars: &'a [std::os::raw::c_char]) -> Self {
        Self::from_chars(chars)
    }
}
impl<'a> From<&'a [f64]> for EValue<'a> {
    fn from(arr: &'a [f64]) -> Self {
        Self::from_f64_arr(arr)
    }
}
impl<'a> From<&'a [bool]> for EValue<'a> {
    fn from(arr: &'a [bool]) -> Self {
        Self::from_bool_arr(arr)
    }
}

impl TryFrom<&EValue<'_>> for i64 {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<i64> {
        match value.tag() {
            Some(Tag::Int) => Ok(unsafe { *value.0.payload.copyable_union.as_int }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl TryFrom<&EValue<'_>> for f64 {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<f64> {
        match value.tag() {
            Some(Tag::Double) => Ok(unsafe { *value.0.payload.copyable_union.as_double }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl TryFrom<&EValue<'_>> for bool {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<bool> {
        match value.tag() {
            Some(Tag::Bool) => Ok(unsafe { *value.0.payload.copyable_union.as_bool }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a, 'b> TryFrom<&'b EValue<'a>> for &'b Tensor<'a> {
    type Error = Error;
    fn try_from(value: &'b EValue<'a>) -> Result<&'b Tensor<'a>> {
        match value.tag() {
            Some(Tag::Tensor) => Ok(unsafe {
                let inner = &*value.0.payload.as_tensor;
                Tensor::from_inner_ref(inner)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<EValue<'a>> for Tensor<'a> {
    type Error = Error;
    fn try_from(mut value: EValue<'a>) -> Result<Tensor<'a>> {
        match value.tag() {
            Some(Tag::Tensor) => Ok(unsafe {
                value.0.tag = et_c::Tag::None;
                let inner = ManuallyDrop::take(&mut value.0.payload.as_tensor);
                Tensor::from_inner(inner)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&EValue<'a>> for &'a [std::os::raw::c_char] {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<&'a [std::os::raw::c_char]> {
        match value.tag() {
            Some(Tag::String) => Ok(unsafe {
                let arr = &*value.0.payload.copyable_union.as_string;
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&EValue<'a>> for &'a [i64] {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<&'a [i64]> {
        match value.tag() {
            Some(Tag::ListInt) => Ok(unsafe {
                let arr = &*value.0.payload.copyable_union.as_int_list;
                BoxedEvalueList::from_inner(arr).get()
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&EValue<'a>> for &'a [f64] {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<&'a [f64]> {
        match value.tag() {
            Some(Tag::ListDouble) => Ok(unsafe {
                let arr = &*value.0.payload.copyable_union.as_double_list;
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&EValue<'a>> for &'a [bool] {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<&'a [bool]> {
        match value.tag() {
            Some(Tag::ListBool) => Ok(unsafe {
                let arr = &*value.0.payload.copyable_union.as_bool_list;
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&EValue<'a>> for &'a [Tensor<'a>] {
    type Error = Error;
    fn try_from(value: &EValue<'_>) -> Result<&'a [Tensor<'a>]> {
        match value.tag() {
            Some(Tag::ListInt) => Ok(unsafe {
                let arr: &et_c::BoxedEvalueList<et_c::Tensor> =
                    &*value.0.payload.copyable_union.as_tensor_list;
                // SAFETY: et_c::Tensor and Tensor have the same memory layout
                let arr = std::mem::transmute::<
                    &et_c::BoxedEvalueList<et_c::Tensor>,
                    &et_c::BoxedEvalueList<Tensor<'a>>,
                >(arr);
                BoxedEvalueList::from_inner(arr).get()
            }),
            _ => Err(Error::InvalidType),
        }
    }
}

/// Helper class used to correlate EValues in the executor table, with the
/// unwrapped list of the proper type. Because values in the runtime's values
/// table can change during execution, we cannot statically allocate list of
/// objects at deserialization. Imagine the serialized list says index 0 in the
/// value table is element 2 in the list, but during execution the value in
/// element 2 changes (in the case of tensor this means the &TensorImpl stored in
/// the tensor changes). To solve this instead they must be created dynamically
/// whenever they are used.
pub struct BoxedEvalueList<'a, T: BoxedEvalue>(et_c::BoxedEvalueList<T>, PhantomData<&'a ()>);
impl<'a, T: BoxedEvalue> BoxedEvalueList<'a, T> {
    pub(crate) unsafe fn from_inner(inner: &et_c::BoxedEvalueList<T>) -> &Self {
        // SAFETY: BoxedEvalueList and et_c::BoxedEvalueList have the same memory layout
        std::mem::transmute::<&et_c::BoxedEvalueList<T>, &BoxedEvalueList<T>>(inner)
    }

    /// Wrapped_vals is a list of pointers into the values table of the runtime
    /// whose destinations correlate with the elements of the list, unwrapped_vals
    /// is a container of the same size whose serves as memory to construct the
    /// unwrapped vals.
    pub fn new(wrapped_vals: &'a [&EValue], unwrapped_vals: &'a mut [T]) -> Self {
        assert_eq!(
            wrapped_vals.len(),
            unwrapped_vals.len(),
            "Length mismatch between wrapped and unwrapped values"
        );
        assert!(
            wrapped_vals.iter().all(|val| val.tag() == Some(T::TAG)),
            "wrapped_vals contains type different from T"
        );

        let wrapped_vals = ArrayRef::from_slice(wrapped_vals).0;
        // SAFETY: EValue and et_c::EValue have the same memory layout
        let wrapped_vals = unsafe {
            std::mem::transmute::<et_c::ArrayRef<&EValue>, et_c::ArrayRef<&et_c::EValue>>(
                wrapped_vals,
            )
        };
        // SAFETY: &et_c::EValue and *mut et_c::EValue have the same memory layout, and ArrayRef is an immutable type
        // so it's safe to transmute its inner values to a mutable type as they will not be mutated
        let wrapped_vals = unsafe {
            std::mem::transmute::<et_c::ArrayRef<&et_c::EValue>, et_c::ArrayRef<*mut et_c::EValue>>(
                wrapped_vals,
            )
        };

        let list = et_c::BoxedEvalueList {
            wrapped_vals_: wrapped_vals,
            unwrapped_vals_: unwrapped_vals.as_mut_ptr(),
            _phantom_0: PhantomData,
        };
        Self(list, PhantomData)
    }

    /// Constructs and returns the list of T specified by the EValue pointers
    pub fn get(&self) -> &'a [T] {
        let evalues = unsafe { ArrayRef::from_inner(&self.0.wrapped_vals_) }.as_slice();
        // SAFETY: EValue and et_c::EValue have the same memory layout
        let evalues = unsafe { std::mem::transmute::<&[*mut et_c::EValue], &[&EValue]>(evalues) };
        assert!(
            evalues.iter().all(|val| val.tag() == Some(T::TAG)),
            "EValues have different tags"
        );

        let unwrapped_list = match T::TAG {
            Tag::Int => {
                // SAFETY: T is i64
                let list = unsafe {
                    std::mem::transmute::<&BoxedEvalueList<'a, T>, &BoxedEvalueList<'a, i64>>(self)
                };
                let unwrapped_list = unsafe { et_rs_c::BoxedEvalueList_i64_get(&list.0) };
                // SAFETY: i64 is T
                unsafe {
                    std::mem::transmute::<et_c::ArrayRef<i64>, et_c::ArrayRef<T>>(unwrapped_list)
                }
            }
            Tag::Tensor => {
                // SAFETY: T is Tensor
                let list = unsafe {
                    std::mem::transmute::<&BoxedEvalueList<'a, T>, &BoxedEvalueList<'a, Tensor<'_>>>(
                        self,
                    )
                };
                // SAFETY: Tensor and et_c::Tensor have the same memory layout
                let list = unsafe {
                    std::mem::transmute::<
                        &et_c::BoxedEvalueList<Tensor<'_>>,
                        &et_c::BoxedEvalueList<et_c::Tensor>,
                    >(&list.0)
                };
                let unwrapped_list = unsafe { et_rs_c::BoxedEvalueList_Tensor_get(list) };
                // SAFETY: et_c::Tensor and Tensor have the same memory layout
                let unwrapped_list = unsafe {
                    std::mem::transmute::<et_c::ArrayRef<et_c::Tensor>, et_c::ArrayRef<Tensor<'_>>>(
                        unwrapped_list,
                    )
                };
                // SAFETY: Tensor is T
                unsafe {
                    std::mem::transmute::<et_c::ArrayRef<Tensor<'a>>, et_c::ArrayRef<T>>(
                        unwrapped_list,
                    )
                }
            }
            unsupported_type => panic!("Unsupported type: {:?}", unsupported_type),
        };
        unsafe { ArrayRef::from_inner(&unwrapped_list).as_slice() }
    }
}

/// A trait for types that can be used within a `BoxedEvalueList`.
pub trait BoxedEvalue {
    const TAG: Tag;
    private_decl! {}
}
impl BoxedEvalue for i64 {
    const TAG: Tag = Tag::Int;
    private_impl! {}
}
impl BoxedEvalue for Tensor<'_> {
    const TAG: Tag = Tag::Tensor;
    private_impl! {}
}
