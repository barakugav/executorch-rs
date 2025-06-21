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

    /// The index is invalid.
    InvalidIndex,
}
impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::CError(error) => std::fmt::Display::fmt(error, fmt),
            Error::ToCStr | Error::FromCStr | Error::InvalidIndex => {
                std::fmt::Debug::fmt(self, fmt)
            }
        }
    }
}
#[cfg(any(error_in_core, feature = "std"))]
impl std::error::Error for Error {}

/// Error codes returned by the Cpp library.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
#[non_exhaustive]
pub enum CError {
    /* System errors */
    //
    /// An internal error occurred.
    Internal = RawCError::Error_Internal as u8,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState = RawCError::Error_InvalidState as u8,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod = RawCError::Error_EndOfMethod as u8,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported = RawCError::Error_NotSupported as u8,
    /// Operation is not yet implemented.
    NotImplemented = RawCError::Error_NotImplemented as u8,
    /// User provided an invalid argument.
    InvalidArgument = RawCError::Error_InvalidArgument as u8,
    /// Object is an invalid type for the operation.
    InvalidType = RawCError::Error_InvalidType as u8,
    /// Operator(s) missing in the operator registry.
    OperatorMissing = RawCError::Error_OperatorMissing as u8,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound = RawCError::Error_NotFound as u8,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed = RawCError::Error_MemoryAllocationFailed as u8,
    /// Could not access a resource.
    AccessFailed = RawCError::Error_AccessFailed as u8,
    /// Error caused by the contents of a program.
    InvalidProgram = RawCError::Error_InvalidProgram as u8,
    /// Error caused by the contents of external data.
    InvalidExternalData = RawCError::Error_InvalidExternalData as u8,
    /// Does not have enough resources to perform the requested operation.
    OutOfResources = RawCError::Error_OutOfResources as u8,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = RawCError::Error_DelegateInvalidCompatibility as u8,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = RawCError::Error_DelegateMemoryAllocationFailed as u8,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = RawCError::Error_DelegateInvalidHandle as u8,
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
            RawCError::Error_InvalidExternalData => CError::InvalidExternalData,
            RawCError::Error_OutOfResources => CError::OutOfResources,
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
    use executorch_sys as et_c;

    use et_c::Error as RawCError;

    use crate::util::IntoRust;

    use super::{CError, Error};

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn cerror_to_error() {
        assert!(matches!(RawCError::Error_Ok.rs(), Ok(())));
        assert!(matches!(
            RawCError::Error_Internal.rs(),
            Err(Error::CError(CError::Internal))
        ));
        assert!(matches!(
            RawCError::Error_InvalidState.rs(),
            Err(Error::CError(CError::InvalidState))
        ));
        assert!(matches!(
            RawCError::Error_EndOfMethod.rs(),
            Err(Error::CError(CError::EndOfMethod))
        ));
        assert!(matches!(
            RawCError::Error_NotSupported.rs(),
            Err(Error::CError(CError::NotSupported))
        ));
        assert!(matches!(
            RawCError::Error_NotImplemented.rs(),
            Err(Error::CError(CError::NotImplemented))
        ));
        assert!(matches!(
            RawCError::Error_InvalidArgument.rs(),
            Err(Error::CError(CError::InvalidArgument))
        ));
        assert!(matches!(
            RawCError::Error_InvalidType.rs(),
            Err(Error::CError(CError::InvalidType))
        ));
        assert!(matches!(
            RawCError::Error_OperatorMissing.rs(),
            Err(Error::CError(CError::OperatorMissing))
        ));
        assert!(matches!(
            RawCError::Error_NotFound.rs(),
            Err(Error::CError(CError::NotFound))
        ));
        assert!(matches!(
            RawCError::Error_MemoryAllocationFailed.rs(),
            Err(Error::CError(CError::MemoryAllocationFailed))
        ));
        assert!(matches!(
            RawCError::Error_AccessFailed.rs(),
            Err(Error::CError(CError::AccessFailed))
        ));
        assert!(matches!(
            RawCError::Error_InvalidProgram.rs(),
            Err(Error::CError(CError::InvalidProgram))
        ));
        assert!(matches!(
            RawCError::Error_InvalidExternalData.rs(),
            Err(Error::CError(CError::InvalidExternalData))
        ));
        assert!(matches!(
            RawCError::Error_OutOfResources.rs(),
            Err(Error::CError(CError::OutOfResources))
        ));
        assert!(matches!(
            RawCError::Error_DelegateInvalidCompatibility.rs(),
            Err(Error::CError(CError::DelegateInvalidCompatibility))
        ));
        assert!(matches!(
            RawCError::Error_DelegateMemoryAllocationFailed.rs(),
            Err(Error::CError(CError::DelegateMemoryAllocationFailed))
        ));
        assert!(matches!(
            RawCError::Error_DelegateInvalidHandle.rs(),
            Err(Error::CError(CError::DelegateInvalidHandle))
        ));
    }
}
