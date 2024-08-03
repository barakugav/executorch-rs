use std::mem::ManuallyDrop;

use crate::{et_c, et_rs_c, util::IntoRust};

/// ExecuTorch Error type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Error {
    /* System errors */
    //
    /// An internal error occurred.
    Internal = et_c::Error::Internal as u8,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState = et_c::Error::InvalidState as u8,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod = et_c::Error::EndOfMethod as u8,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported = et_c::Error::NotSupported as u8,
    /// Operation is not yet implemented.
    NotImplemented = et_c::Error::NotImplemented as u8,
    /// User provided an invalid argument.
    InvalidArgument = et_c::Error::InvalidArgument as u8,
    /// Object is an invalid type for the operation.
    InvalidType = et_c::Error::InvalidType as u8,
    /// Operator(s) missing in the operator registry.
    OperatorMissing = et_c::Error::OperatorMissing as u8,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound = et_c::Error::NotFound as u8,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed = et_c::Error::MemoryAllocationFailed as u8,
    /// Could not access a resource.
    AccessFailed = et_c::Error::AccessFailed as u8,
    /// Error caused by the contents of a program.
    InvalidProgram = et_c::Error::InvalidProgram as u8,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = et_c::Error::DelegateInvalidCompatibility as u8,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = et_c::Error::DelegateMemoryAllocationFailed as u8,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = et_c::Error::DelegateInvalidHandle as u8,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Error::Internal => "An internal error occurred",
            Error::InvalidState => "Executor is in an invalid state for a target",
            Error::EndOfMethod => "No more steps of execution to run",
            Error::NotSupported => "Operation is not supported in the current context",
            Error::NotImplemented => "Operation is not yet implemented",
            Error::InvalidArgument => "User provided an invalid argument",
            Error::InvalidType => "Object is an invalid type for the operation",
            Error::OperatorMissing => "Operator(s) missing in the operator registry",
            Error::NotFound => "Requested resource could not be found",
            Error::MemoryAllocationFailed => "Could not allocate the requested memory",
            Error::AccessFailed => "Could not access a resource",
            Error::InvalidProgram => "Error caused by the contents of a program",
            Error::DelegateInvalidCompatibility => {
                "Backend receives an incompatible delegate version"
            }
            Error::DelegateMemoryAllocationFailed => "Backend fails to allocate memory",
            Error::DelegateInvalidHandle => "The handle is invalid",
        };
        write!(f, "{}", msg)
    }
}
impl std::error::Error for Error {}

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

#[cfg(test)]
mod tests {
    use crate::Error;

    #[test]
    fn test_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Error>();
    }

    #[test]
    fn test_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Error>();
    }
}
