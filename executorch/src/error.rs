//! Error types used in the [`executortorch`](crate) crate.

use crate::util::IntoRust;
use crate::sys;

/// ExecuTorch Error type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
#[non_exhaustive]
pub enum Error {
    /* System errors */
    //
    /// An internal error occurred.
    Internal = sys::Error::Error_Internal as u32,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState = sys::Error::Error_InvalidState as u32,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod = sys::Error::Error_EndOfMethod as u32,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported = sys::Error::Error_NotSupported as u32,
    /// Operation is not yet implemented.
    NotImplemented = sys::Error::Error_NotImplemented as u32,
    /// User provided an invalid argument.
    InvalidArgument = sys::Error::Error_InvalidArgument as u32,
    /// Object is an invalid type for the operation.
    InvalidType = sys::Error::Error_InvalidType as u32,
    /// Operator(s) missing in the operator registry.
    OperatorMissing = sys::Error::Error_OperatorMissing as u32,
    /// Registration error: Exceeding the maximum number of kernels.
    RegistrationExceedingMaxKernels = sys::Error::Error_RegistrationExceedingMaxKernels as u32,
    /// Registration error: The kernel is already registered.
    RegistrationAlreadyRegistered = sys::Error::Error_RegistrationAlreadyRegistered as u32,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound = sys::Error::Error_NotFound as u32,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed = sys::Error::Error_MemoryAllocationFailed as u32,
    /// Could not access a resource.
    AccessFailed = sys::Error::Error_AccessFailed as u32,
    /// Error caused by the contents of a program.
    InvalidProgram = sys::Error::Error_InvalidProgram as u32,
    /// Error caused by the contents of external data.
    InvalidExternalData = sys::Error::Error_InvalidExternalData as u32,
    /// Does not have enough resources to perform the requested operation.
    OutOfResources = sys::Error::Error_OutOfResources as u32,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = sys::Error::Error_DelegateInvalidCompatibility as u32,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = sys::Error::Error_DelegateMemoryAllocationFailed as u32,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = sys::Error::Error_DelegateInvalidHandle as u32,

    /// Invalid string.
    ///
    /// Error used for example when an invalid UTF-8 is encountered when converting a CStr to a Rust &str, or when
    /// a &str contains null bytes when converting to a CStr, etc.
    InvalidString,
}
impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, fmt)
    }
}
#[cfg(any(error_in_core, feature = "std"))]
impl std::error::Error for Error {}

impl IntoRust for sys::Error {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(match self {
            sys::Error::Error_Ok => return Ok(()),
            sys::Error::Error_Internal => Error::Internal,
            sys::Error::Error_InvalidState => Error::InvalidState,
            sys::Error::Error_EndOfMethod => Error::EndOfMethod,
            sys::Error::Error_NotSupported => Error::NotSupported,
            sys::Error::Error_NotImplemented => Error::NotImplemented,
            sys::Error::Error_InvalidArgument => Error::InvalidArgument,
            sys::Error::Error_InvalidType => Error::InvalidType,
            sys::Error::Error_OperatorMissing => Error::OperatorMissing,
            sys::Error::Error_RegistrationExceedingMaxKernels => {
                Error::RegistrationExceedingMaxKernels
            }
            sys::Error::Error_RegistrationAlreadyRegistered => Error::RegistrationAlreadyRegistered,
            sys::Error::Error_NotFound => Error::NotFound,
            sys::Error::Error_MemoryAllocationFailed => Error::MemoryAllocationFailed,
            sys::Error::Error_AccessFailed => Error::AccessFailed,
            sys::Error::Error_InvalidProgram => Error::InvalidProgram,
            sys::Error::Error_InvalidExternalData => Error::InvalidExternalData,
            sys::Error::Error_OutOfResources => Error::OutOfResources,
            sys::Error::Error_DelegateInvalidCompatibility => Error::DelegateInvalidCompatibility,
            sys::Error::Error_DelegateMemoryAllocationFailed => {
                Error::DelegateMemoryAllocationFailed
            }
            sys::Error::Error_DelegateInvalidHandle => Error::DelegateInvalidHandle,
        })
    }
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use crate::sys;

    use crate::util::IntoRust;

    use super::Error;

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn cerror_to_error() {
        assert!(matches!(sys::Error::Error_Ok.rs(), Ok(())));
        assert!(matches!(
            sys::Error::Error_Internal.rs(),
            Err(Error::Internal)
        ));
        assert!(matches!(
            sys::Error::Error_InvalidState.rs(),
            Err(Error::InvalidState)
        ));
        assert!(matches!(
            sys::Error::Error_EndOfMethod.rs(),
            Err(Error::EndOfMethod)
        ));
        assert!(matches!(
            sys::Error::Error_NotSupported.rs(),
            Err(Error::NotSupported)
        ));
        assert!(matches!(
            sys::Error::Error_NotImplemented.rs(),
            Err(Error::NotImplemented)
        ));
        assert!(matches!(
            sys::Error::Error_InvalidArgument.rs(),
            Err(Error::InvalidArgument)
        ));
        assert!(matches!(
            sys::Error::Error_InvalidType.rs(),
            Err(Error::InvalidType)
        ));
        assert!(matches!(
            sys::Error::Error_OperatorMissing.rs(),
            Err(Error::OperatorMissing)
        ));
        assert!(matches!(
            sys::Error::Error_NotFound.rs(),
            Err(Error::NotFound)
        ));
        assert!(matches!(
            sys::Error::Error_MemoryAllocationFailed.rs(),
            Err(Error::MemoryAllocationFailed)
        ));
        assert!(matches!(
            sys::Error::Error_AccessFailed.rs(),
            Err(Error::AccessFailed)
        ));
        assert!(matches!(
            sys::Error::Error_InvalidProgram.rs(),
            Err(Error::InvalidProgram)
        ));
        assert!(matches!(
            sys::Error::Error_InvalidExternalData.rs(),
            Err(Error::InvalidExternalData)
        ));
        assert!(matches!(
            sys::Error::Error_OutOfResources.rs(),
            Err(Error::OutOfResources)
        ));
        assert!(matches!(
            sys::Error::Error_DelegateInvalidCompatibility.rs(),
            Err(Error::DelegateInvalidCompatibility)
        ));
        assert!(matches!(
            sys::Error::Error_DelegateMemoryAllocationFailed.rs(),
            Err(Error::DelegateMemoryAllocationFailed)
        ));
        assert!(matches!(
            sys::Error::Error_DelegateInvalidHandle.rs(),
            Err(Error::DelegateInvalidHandle)
        ));
    }
}
