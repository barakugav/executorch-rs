//! Error types used in the [`executortorch`](crate) crate.

use core::mem::MaybeUninit;

use crate::{et_c, util::IntoRust};

use et_c::runtime::Error as CError;

/// ExecuTorch Error type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Error {
    /* System errors */
    //
    /// An internal error occurred.
    Internal,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported,
    /// Operation is not yet implemented.
    NotImplemented,
    /// User provided an invalid argument.
    InvalidArgument,
    /// Object is an invalid type for the operation.
    InvalidType,
    /// Operator(s) missing in the operator registry.
    OperatorMissing,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed,
    /// Could not access a resource.
    AccessFailed,
    /// Error caused by the contents of a program.
    InvalidProgram,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle,
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
