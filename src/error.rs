//! Error types used in the [`executortorch`](crate) crate.

use core::mem::MaybeUninit;

use crate::{et_c, util::IntoRust};

/// ExecuTorch Error type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Error {
    /* System errors */
    //
    /// An internal error occurred.
    Internal = et_c::runtime::Error::Internal as u8,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState = et_c::runtime::Error::InvalidState as u8,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod = et_c::runtime::Error::EndOfMethod as u8,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported = et_c::runtime::Error::NotSupported as u8,
    /// Operation is not yet implemented.
    NotImplemented = et_c::runtime::Error::NotImplemented as u8,
    /// User provided an invalid argument.
    InvalidArgument = et_c::runtime::Error::InvalidArgument as u8,
    /// Object is an invalid type for the operation.
    InvalidType = et_c::runtime::Error::InvalidType as u8,
    /// Operator(s) missing in the operator registry.
    OperatorMissing = et_c::runtime::Error::OperatorMissing as u8,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound = et_c::runtime::Error::NotFound as u8,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed = et_c::runtime::Error::MemoryAllocationFailed as u8,
    /// Could not access a resource.
    AccessFailed = et_c::runtime::Error::AccessFailed as u8,
    /// Error caused by the contents of a program.
    InvalidProgram = et_c::runtime::Error::InvalidProgram as u8,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = et_c::runtime::Error::DelegateInvalidCompatibility as u8,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = et_c::runtime::Error::DelegateMemoryAllocationFailed as u8,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = et_c::runtime::Error::DelegateInvalidHandle as u8,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl IntoRust for et_c::runtime::Error {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(match self {
            et_c::runtime::Error::Ok => return Ok(()),
            et_c::runtime::Error::Internal => Error::Internal,
            et_c::runtime::Error::InvalidState => Error::InvalidState,
            et_c::runtime::Error::EndOfMethod => Error::EndOfMethod,
            et_c::runtime::Error::NotSupported => Error::NotSupported,
            et_c::runtime::Error::NotImplemented => Error::NotImplemented,
            et_c::runtime::Error::InvalidArgument => Error::InvalidArgument,
            et_c::runtime::Error::InvalidType => Error::InvalidType,
            et_c::runtime::Error::OperatorMissing => Error::OperatorMissing,
            et_c::runtime::Error::NotFound => Error::NotFound,
            et_c::runtime::Error::MemoryAllocationFailed => Error::MemoryAllocationFailed,
            et_c::runtime::Error::AccessFailed => Error::AccessFailed,
            et_c::runtime::Error::InvalidProgram => Error::InvalidProgram,
            et_c::runtime::Error::DelegateInvalidCompatibility => {
                Error::DelegateInvalidCompatibility
            }
            et_c::runtime::Error::DelegateMemoryAllocationFailed => {
                Error::DelegateMemoryAllocationFailed
            }
            et_c::runtime::Error::DelegateInvalidHandle => Error::DelegateInvalidHandle,
        })
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn fallible<T>(f: impl FnOnce(*mut T) -> et_c::runtime::Error) -> Result<T> {
    let mut value = MaybeUninit::uninit();
    let err = f(value.as_mut_ptr());
    err.rs().map(|_| unsafe { value.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::Error;

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
