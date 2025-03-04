//! Error types used in the [`executortorch`](crate) crate.

use crate::util::IntoRust;
use executorch_sys as et_c;

use et_c::Error as RawCError;

/// ExecuTorch Error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An error from the Cpp underlying library.
    CError(CError),

    /// Failed to convert a string or Path to a CStr.
    ToCStr,

    /// Failed to convert from CStr to str.
    FromCStr,

    /// Failed to allocate memory.
    AllocationFailed,

    /// The index is invalid.
    InvalidIndex,
}
impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::CError(error) => std::fmt::Display::fmt(error, fmt),
            Error::ToCStr | Error::FromCStr | Error::AllocationFailed | Error::InvalidIndex => {
                std::fmt::Debug::fmt(self, fmt)
            }
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
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, fmt)
    }
}
#[cfg(feature = "std")]
impl std::error::Error for CError {}

impl IntoRust for RawCError {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(Error::CError(match self {
            RawCError::Error_Ok => return Ok(()),
            RawCError::Error_Internal => CError::Internal,
            RawCError::Error_InvalidState => CError::InvalidState,
            RawCError::Error_EndOfMethod => CError::EndOfMethod,
            RawCError::Error_NotSupported => CError::NotSupported,
            RawCError::Error_NotImplemented => CError::NotImplemented,
            RawCError::Error_InvalidArgument => CError::InvalidArgument,
            RawCError::Error_InvalidType => CError::InvalidType,
            RawCError::Error_OperatorMissing => CError::OperatorMissing,
            RawCError::Error_NotFound => CError::NotFound,
            RawCError::Error_MemoryAllocationFailed => CError::MemoryAllocationFailed,
            RawCError::Error_AccessFailed => CError::AccessFailed,
            RawCError::Error_InvalidProgram => CError::InvalidProgram,
            RawCError::Error_DelegateInvalidCompatibility => CError::DelegateInvalidCompatibility,
            RawCError::Error_DelegateMemoryAllocationFailed => {
                CError::DelegateMemoryAllocationFailed
            }
            RawCError::Error_DelegateInvalidHandle => CError::DelegateInvalidHandle,
        }))
    }
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }
}
