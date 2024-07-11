use std::mem::ManuallyDrop;

use crate::{et_c, et_rs_c, util::IntoRust};

#[derive(Debug)]
pub enum Error {
    Internal,
    InvalidState,
    EndOfMethod,
    NotSupported,
    NotImplemented,
    InvalidArgument,
    InvalidType,
    OperatorMissing,
    NotFound,
    MemoryAllocationFailed,
    AccessFailed,
    InvalidProgram,
    DelegateInvalidCompatibility,
    DelegateMemoryAllocationFailed,
    DelegateInvalidHandle,
}

impl IntoRust for et_c::Error {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(match self {
            et_c::Error_Ok => return Ok(()),
            et_c::Error_Internal => Error::Internal,
            et_c::Error_InvalidState => Error::InvalidState,
            et_c::Error_EndOfMethod => Error::EndOfMethod,
            et_c::Error_NotSupported => Error::NotSupported,
            et_c::Error_NotImplemented => Error::NotImplemented,
            et_c::Error_InvalidArgument => Error::InvalidArgument,
            et_c::Error_InvalidType => Error::InvalidType,
            et_c::Error_OperatorMissing => Error::OperatorMissing,
            et_c::Error_NotFound => Error::NotFound,
            et_c::Error_MemoryAllocationFailed => Error::MemoryAllocationFailed,
            et_c::Error_AccessFailed => Error::AccessFailed,
            et_c::Error_InvalidProgram => Error::InvalidProgram,
            et_c::Error_DelegateInvalidCompatibility => Error::DelegateInvalidCompatibility,
            et_c::Error_DelegateMemoryAllocationFailed => Error::DelegateMemoryAllocationFailed,
            et_c::Error_DelegateInvalidHandle => Error::DelegateInvalidHandle,
            unknown_err => {
                log::debug!("Unknown error {unknown_err}");
                Error::Internal
            }
        })
    }
}

fn to_bytes<T>(val: &T) -> Vec<u8> {
    (0..std::mem::size_of_val(val))
        .map(|i| unsafe {
            let ptr = val as *const _;
            let ptr = ptr as usize;
            let ptr = ptr as *const u8;
            *ptr.add(i)
        })
        .collect()
}

pub type Result<T> = std::result::Result<T, Error>;
impl<T> IntoRust for et_c::Result<T> {
    type RsType = Result<T>;
    fn rs(self) -> Self::RsType {
        if self.hasValue_ {
            let value = unsafe { ManuallyDrop::into_inner(self.__bindgen_anon_1.value_) };
            Ok(value)
        } else {
            println!("{:?}", to_bytes(&self));

            let err: et_c::Error =
                unsafe { ManuallyDrop::into_inner(self.__bindgen_anon_1.error_) };
            Err(err.rs().err().unwrap_or_else(|| {
                log::debug!("Error_Ok should not happen");
                Error::Internal
            }))
        }
    }
}
impl IntoRust for et_rs_c::Result_i64 {
    type RsType = Result<i64>;
    fn rs(self) -> Self::RsType {
        if self.hasValue_ {
            let value = unsafe { ManuallyDrop::into_inner(self.__bindgen_anon_1.value_) };
            Ok(value)
        } else {
            let err: et_c::Error =
                unsafe { ManuallyDrop::into_inner(self.__bindgen_anon_1.error_) };
            Err(err.rs().err().unwrap_or_else(|| {
                log::debug!("Error_Ok should not happen");
                Error::Internal
            }))
        }
    }
}
impl IntoRust for et_rs_c::Result_MethodMeta {
    type RsType = Result<et_c::MethodMeta>;
    fn rs(self) -> Self::RsType {
        if self.hasValue_ {
            let value = unsafe { ManuallyDrop::into_inner(self.__bindgen_anon_1.value_) };
            Ok(value)
        } else {
            let err: et_c::Error =
                unsafe { ManuallyDrop::into_inner(self.__bindgen_anon_1.error_) };
            Err(err.rs().err().unwrap_or_else(|| {
                log::debug!("Error_Ok should not happen");
                Error::Internal
            }))
        }
    }
}
