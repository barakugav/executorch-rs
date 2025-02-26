//! Error types used in the [`executortorch`](crate) crate.

use std::mem::MaybeUninit;

use crate::{et_c, util::IntoRust};

use et_c::runtime::Error as RawCError;

/// ExecuTorch Error type.
#[derive(Debug)]
pub enum Error {
    /// An error from the Cpp underlying library.
    CError(CError),
}
impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::CError(error) => std::fmt::Display::fmt(error, fmt),
        }
    }
}
#[cfg(any(error_in_core, feature = "std"))]
impl std::error::Error for Error {}

/// Categories of errors that can occur in executorch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum CError {
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
impl std::fmt::Display for CError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for CError {}

impl IntoRust for RawCError {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(Error::CError(match self {
            RawCError::Ok => return Ok(()),
            RawCError::Internal => CError::Internal,
            RawCError::InvalidState => CError::InvalidState,
            RawCError::EndOfMethod => CError::EndOfMethod,
            RawCError::NotSupported => CError::NotSupported,
            RawCError::NotImplemented => CError::NotImplemented,
            RawCError::InvalidArgument => CError::InvalidArgument,
            RawCError::InvalidType => CError::InvalidType,
            RawCError::OperatorMissing => CError::OperatorMissing,
            RawCError::NotFound => CError::NotFound,
            RawCError::MemoryAllocationFailed => CError::MemoryAllocationFailed,
            RawCError::AccessFailed => CError::AccessFailed,
            RawCError::InvalidProgram => CError::InvalidProgram,
            RawCError::DelegateInvalidCompatibility => CError::DelegateInvalidCompatibility,
            RawCError::DelegateMemoryAllocationFailed => CError::DelegateMemoryAllocationFailed,
            RawCError::DelegateInvalidHandle => CError::DelegateInvalidHandle,
        }))
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn try_new<T>(f: impl FnOnce(*mut T) -> RawCError) -> crate::Result<T> {
    let mut value = MaybeUninit::uninit();
    let err = f(value.as_mut_ptr());
    err.rs().map(|_| unsafe { value.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }
}
