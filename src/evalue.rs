use std::{marker::PhantomData, mem::ManuallyDrop};

use crate::{et_c, tensor::Tensor, Error, Result};

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Tag {
    Tensor = et_c::Tag::Tensor as u8,
    String = et_c::Tag::String as u8,
    Double = et_c::Tag::Double as u8,
    Int = et_c::Tag::Int as u8,
    Bool = et_c::Tag::Bool as u8,
    ListBool = et_c::Tag::ListBool as u8,
    ListDouble = et_c::Tag::ListDouble as u8,
    ListInt = et_c::Tag::ListInt as u8,
    ListTensor = et_c::Tag::ListTensor as u8,
    ListScalar = et_c::Tag::ListScalar as u8,
    ListOptionalTensor = et_c::Tag::ListOptionalTensor as u8,
}

pub struct EValue<'a>(et_c::EValue, PhantomData<&'a ()>);
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

    pub fn from_i64(val: i64) -> EValue<'static> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_int: ManuallyDrop::new(val),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::Int) }
    }

    pub fn from_f64(val: f64) -> EValue<'static> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_double: ManuallyDrop::new(val),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::Double) }
    }

    pub fn from_bool(val: bool) -> EValue<'static> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_bool: ManuallyDrop::new(val),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::Bool) }
    }

    pub fn from_chars(chars: &'a [std::os::raw::c_char]) -> EValue<'a> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_string: ManuallyDrop::new(unsafe { array_ref(chars) }),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::String) }
    }

    pub fn from_f64_arr(arr: &'a [f64]) -> EValue<'a> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_double_list: ManuallyDrop::new(unsafe { array_ref(arr) }),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListDouble) }
    }

    pub fn from_bool_arr(arr: &'a [bool]) -> EValue<'a> {
        let value = et_c::EValue_Payload_TriviallyCopyablePayload {
            as_bool_list: ManuallyDrop::new(unsafe { array_ref(arr) }),
        };
        unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListBool) }
    }

    // pub fn from_i64_arr(arr: &'a [i64], temp_arr: &'a mut [i64]) -> Self {
    //     let value = EValue_Payload_TriviallyCopyablePayload {
    //         as_int_list: ManuallyDrop::new(et_c::ArrayRef {
    //             Data: self.as_ptr(),
    //             Length: self.len(),
    //             _phantom_0: PhantomData,
    //         }),
    //     };
    //     unsafe { EValue::new_trivially_copyable(value, et_c::Tag::ListInt) }
    // }

    pub fn from_tensor(tensor: Tensor<'a>) -> EValue<'a> {
        unsafe {
            EValue::new(et_c::EValue {
                payload: et_c::EValue_Payload {
                    as_tensor: ManuallyDrop::new(tensor.into_inner()),
                },
                tag: et_c::Tag::Tensor,
            })
        }
    }

    #[track_caller]
    pub fn into_i64(self) -> i64 {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn into_f64(self) -> f64 {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn into_bool(self) -> bool {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn into_chars(self) -> &'a [std::os::raw::c_char] {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn into_f64_arr(self) -> &'a [f64] {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn into_bool_arr(self) -> &'a [bool] {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn into_tensor(self) -> Tensor<'a> {
        self.try_into().expect("Invalid type")
    }

    #[track_caller]
    pub fn as_tensor(&self) -> &Tensor<'a> {
        self.try_into().expect("Invalid type")
    }

    pub fn tag(&self) -> Option<Tag> {
        Some(match self.0.tag {
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
            et_c::Tag::ListScalar => Tag::ListScalar,
            et_c::Tag::ListOptionalTensor => Tag::ListOptionalTensor,
        })
    }

    pub(crate) fn inner(&self) -> &et_c::EValue {
        &self.0
    }

    // pub(crate) fn into_inner(self) -> et_c::EValue {
    //     self.0
    // }
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

impl TryFrom<EValue<'_>> for i64 {
    type Error = Error;
    fn try_from(value: EValue<'_>) -> Result<i64> {
        match value.tag() {
            Some(Tag::Int) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.copyable_union);
                ManuallyDrop::into_inner(inner.as_int)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl TryFrom<EValue<'_>> for f64 {
    type Error = Error;
    fn try_from(value: EValue<'_>) -> Result<f64> {
        match value.tag() {
            Some(Tag::Double) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.copyable_union);
                ManuallyDrop::into_inner(inner.as_double)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl TryFrom<EValue<'_>> for bool {
    type Error = Error;
    fn try_from(value: EValue<'_>) -> Result<bool> {
        match value.tag() {
            Some(Tag::Bool) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.copyable_union);
                ManuallyDrop::into_inner(inner.as_bool)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<EValue<'a>> for &'a [std::os::raw::c_char] {
    type Error = Error;
    fn try_from(value: EValue<'_>) -> Result<&'_ [std::os::raw::c_char]> {
        match value.tag() {
            Some(Tag::String) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.copyable_union);
                let arr = ManuallyDrop::into_inner(inner.as_string);
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<EValue<'a>> for &'a [f64] {
    type Error = Error;
    fn try_from(value: EValue<'_>) -> Result<&'_ [f64]> {
        match value.tag() {
            Some(Tag::ListDouble) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.copyable_union);
                let arr = ManuallyDrop::into_inner(inner.as_double_list);
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<EValue<'a>> for &'a [bool] {
    type Error = Error;
    fn try_from(value: EValue<'_>) -> Result<&'_ [bool]> {
        match value.tag() {
            Some(Tag::ListBool) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.copyable_union);
                let arr = ManuallyDrop::into_inner(inner.as_bool_list);
                std::slice::from_raw_parts(arr.Data, arr.Length)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<EValue<'a>> for Tensor<'a> {
    type Error = Error;
    fn try_from(value: EValue<'a>) -> Result<Tensor<'a>> {
        match value.tag() {
            Some(Tag::Tensor) => Ok(unsafe {
                let inner = ManuallyDrop::into_inner(value.0.payload.as_tensor);
                Tensor::from_inner(inner)
            }),
            _ => Err(Error::InvalidType),
        }
    }
}
impl<'a> TryFrom<&EValue<'a>> for &Tensor<'a> {
    type Error = Error;
    fn try_from(value: &EValue<'a>) -> Result<&'a Tensor<'a>> {
        match value.tag() {
            Some(Tag::Tensor) => Ok(unsafe {
                let inner = &*value.0.payload.as_tensor;
                // SAFETY: et_c::Tensor has the same memory layout as Tensor
                let ptr = inner as *const et_c::Tensor as *const Tensor<'a>;
                &*ptr
            }),
            _ => Err(Error::InvalidType),
        }
    }
}

unsafe fn array_ref<T>(s: &[T]) -> et_c::ArrayRef<T> {
    et_c::ArrayRef {
        Data: s.as_ptr(),
        Length: s.len(),
        _phantom_0: PhantomData,
    }
}

// pub struct BoxedEvalueList<'a, T>(et_c::BoxedEvalueList<T>, PhantomData<&'a ()>);
// impl<'a, T> BoxedEvalueList<'a, T> {
//     pub fn new(wrapped_vals: &'a [T], unwrapped_vals: &'a mut [T]) -> Self {
//         Self(
//             et_c::BoxedEvalueList {
//                 wrapped_vals_: unsafe { array_ref(wrapped_vals) },
//                 unwrapped_vals_: unwrapped_vals,
//                 _phantom_0: PhantomData,
//             },
//             PhantomData,
//         )
//     }
// }
