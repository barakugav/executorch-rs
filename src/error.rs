//! Error types used in the [`executortorch`](crate) crate.

use core::mem::MaybeUninit;

use crate::{et_c, util::IntoRust};

use et_c::runtime::Error as CError;

/// ExecuTorch Error type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Error {
    /* System errors */
    //
    /// An internal error occurred.
    Internal = CError::Internal as u8,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState = CError::InvalidState as u8,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod = CError::EndOfMethod as u8,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported = CError::NotSupported as u8,
    /// Operation is not yet implemented.
    NotImplemented = CError::NotImplemented as u8,
    /// User provided an invalid argument.
    InvalidArgument = CError::InvalidArgument as u8,
    /// Object is an invalid type for the operation.
    InvalidType = CError::InvalidType as u8,
    /// Operator(s) missing in the operator registry.
    OperatorMissing = CError::OperatorMissing as u8,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound = CError::NotFound as u8,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed = CError::MemoryAllocationFailed as u8,
    /// Could not access a resource.
    AccessFailed = CError::AccessFailed as u8,
    /// Error caused by the contents of a program.
    InvalidProgram = CError::InvalidProgram as u8,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = CError::DelegateInvalidCompatibility as u8,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = CError::DelegateMemoryAllocationFailed as u8,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = CError::DelegateInvalidHandle as u8,
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

impl IntoRust for CError {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(match self {
            CError::Ok => return Ok(()),
            CError::Internal => Error::Internal,
            CError::InvalidState => Error::InvalidState,
            CError::EndOfMethod => Error::EndOfMethod,
            CError::NotSupported => Error::NotSupported,
            CError::NotImplemented => Error::NotImplemented,
            CError::InvalidArgument => Error::InvalidArgument,
            CError::InvalidType => Error::InvalidType,
            CError::OperatorMissing => Error::OperatorMissing,
            CError::NotFound => Error::NotFound,
            CError::MemoryAllocationFailed => Error::MemoryAllocationFailed,
            CError::AccessFailed => Error::AccessFailed,
            CError::InvalidProgram => Error::InvalidProgram,
            CError::DelegateInvalidCompatibility => Error::DelegateInvalidCompatibility,
            CError::DelegateMemoryAllocationFailed => Error::DelegateMemoryAllocationFailed,
            CError::DelegateInvalidHandle => Error::DelegateInvalidHandle,
        })
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn fallible<T>(f: impl FnOnce(*mut T) -> CError) -> Result<T> {
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
