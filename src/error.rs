use std::mem::ManuallyDrop;

use crate::{et_c, et_rs_c, util::IntoRust};

#[derive(Debug)]
#[repr(u8)]
pub enum Error {
    Internal = et_c::Error::Internal as u8,
    InvalidState = et_c::Error::InvalidState as u8,
    EndOfMethod = et_c::Error::EndOfMethod as u8,
    NotSupported = et_c::Error::NotSupported as u8,
    NotImplemented = et_c::Error::NotImplemented as u8,
    InvalidArgument = et_c::Error::InvalidArgument as u8,
    InvalidType = et_c::Error::InvalidType as u8,
    OperatorMissing = et_c::Error::OperatorMissing as u8,
    NotFound = et_c::Error::NotFound as u8,
    MemoryAllocationFailed = et_c::Error::MemoryAllocationFailed as u8,
    AccessFailed = et_c::Error::AccessFailed as u8,
    InvalidProgram = et_c::Error::InvalidProgram as u8,
    DelegateInvalidCompatibility = et_c::Error::DelegateInvalidCompatibility as u8,
    DelegateMemoryAllocationFailed = et_c::Error::DelegateMemoryAllocationFailed as u8,
    DelegateInvalidHandle = et_c::Error::DelegateInvalidHandle as u8,
}

impl IntoRust for et_c::Error {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(match self {
            et_c::Error::Ok => return Ok(()),
            et_c::Error::Internal => Error::Internal,
            et_c::Error::InvalidState => Error::InvalidState,
            et_c::Error::EndOfMethod => Error::EndOfMethod,
            et_c::Error::NotSupported => Error::NotSupported,
            et_c::Error::NotImplemented => Error::NotImplemented,
            et_c::Error::InvalidArgument => Error::InvalidArgument,
            et_c::Error::InvalidType => Error::InvalidType,
            et_c::Error::OperatorMissing => Error::OperatorMissing,
            et_c::Error::NotFound => Error::NotFound,
            et_c::Error::MemoryAllocationFailed => Error::MemoryAllocationFailed,
            et_c::Error::AccessFailed => Error::AccessFailed,
            et_c::Error::InvalidProgram => Error::InvalidProgram,
            et_c::Error::DelegateInvalidCompatibility => Error::DelegateInvalidCompatibility,
            et_c::Error::DelegateMemoryAllocationFailed => Error::DelegateMemoryAllocationFailed,
            et_c::Error::DelegateInvalidHandle => Error::DelegateInvalidHandle,
        })
    }
}

pub type Result<T> = std::result::Result<T, Error>;
impl<T> IntoRust for et_c::Result<T> {
    type RsType = Result<T>;
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
