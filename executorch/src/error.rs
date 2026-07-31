//! Error types used in the [`executortorch`](crate) crate.

use crate::util::IntoRust;
use executorch_sys::ET_Error;

/// ExecuTorch Error type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
#[non_exhaustive]
pub enum Error {
    /* System errors */
    //
    /// An internal error occurred.
    Internal = ET_Error::ET_Error_Internal as u32,
    /// Status indicating the executor is in an invalid state for a target
    InvalidState = ET_Error::ET_Error_InvalidState as u32,
    /// Status indicating there are no more steps of execution to run
    EndOfMethod = ET_Error::ET_Error_EndOfMethod as u32,
    /// Status indicating a resource has already been loaded.
    AlreadyLoaded = ET_Error::ET_Error_AlreadyLoaded as u32,

    /* Logical errors */
    //
    /// Operation is not supported in the current context.
    NotSupported = ET_Error::ET_Error_NotSupported as u32,
    /// Operation is not yet implemented.
    NotImplemented = ET_Error::ET_Error_NotImplemented as u32,
    /// User provided an invalid argument.
    InvalidArgument = ET_Error::ET_Error_InvalidArgument as u32,
    /// Object is an invalid type for the operation.
    InvalidType = ET_Error::ET_Error_InvalidType as u32,
    /// Operator(s) missing in the operator registry.
    OperatorMissing = ET_Error::ET_Error_OperatorMissing as u32,
    /// Registration error: Exceeding the maximum number of kernels.
    RegistrationExceedingMaxKernels = ET_Error::ET_Error_RegistrationExceedingMaxKernels as u32,
    /// Registration error: The kernel is already registered.
    RegistrationAlreadyRegistered = ET_Error::ET_Error_RegistrationAlreadyRegistered as u32,

    /* Resource errors */
    //
    /// Requested resource could not be found.
    NotFound = ET_Error::ET_Error_NotFound as u32,
    /// Could not allocate the requested memory.
    MemoryAllocationFailed = ET_Error::ET_Error_MemoryAllocationFailed as u32,
    /// Could not access a resource.
    AccessFailed = ET_Error::ET_Error_AccessFailed as u32,
    /// Error caused by the contents of a program.
    InvalidProgram = ET_Error::ET_Error_InvalidProgram as u32,
    /// Error caused by the contents of external data.
    InvalidExternalData = ET_Error::ET_Error_InvalidExternalData as u32,
    /// Does not have enough resources to perform the requested operation.
    OutOfResources = ET_Error::ET_Error_OutOfResources as u32,

    /* Delegate errors */
    //
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = ET_Error::ET_Error_DelegateInvalidCompatibility as u32,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = ET_Error::ET_Error_DelegateMemoryAllocationFailed as u32,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = ET_Error::ET_Error_DelegateInvalidHandle as u32,

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

impl IntoRust for ET_Error {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        Err(match self {
            ET_Error::ET_Error_Ok => return Ok(()),
            ET_Error::ET_Error_Internal => Error::Internal,
            ET_Error::ET_Error_InvalidState => Error::InvalidState,
            ET_Error::ET_Error_EndOfMethod => Error::EndOfMethod,
            ET_Error::ET_Error_AlreadyLoaded => Error::AlreadyLoaded,
            ET_Error::ET_Error_NotSupported => Error::NotSupported,
            ET_Error::ET_Error_NotImplemented => Error::NotImplemented,
            ET_Error::ET_Error_InvalidArgument => Error::InvalidArgument,
            ET_Error::ET_Error_InvalidType => Error::InvalidType,
            ET_Error::ET_Error_OperatorMissing => Error::OperatorMissing,
            ET_Error::ET_Error_RegistrationExceedingMaxKernels => {
                Error::RegistrationExceedingMaxKernels
            }
            ET_Error::ET_Error_RegistrationAlreadyRegistered => {
                Error::RegistrationAlreadyRegistered
            }
            ET_Error::ET_Error_NotFound => Error::NotFound,
            ET_Error::ET_Error_MemoryAllocationFailed => Error::MemoryAllocationFailed,
            ET_Error::ET_Error_AccessFailed => Error::AccessFailed,
            ET_Error::ET_Error_InvalidProgram => Error::InvalidProgram,
            ET_Error::ET_Error_InvalidExternalData => Error::InvalidExternalData,
            ET_Error::ET_Error_OutOfResources => Error::OutOfResources,
            ET_Error::ET_Error_DelegateInvalidCompatibility => Error::DelegateInvalidCompatibility,
            ET_Error::ET_Error_DelegateMemoryAllocationFailed => {
                Error::DelegateMemoryAllocationFailed
            }
            ET_Error::ET_Error_DelegateInvalidHandle => Error::DelegateInvalidHandle,
        })
    }
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use executorch_sys::ET_Error;

    use crate::util::IntoRust;

    use super::Error;

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn cerror_to_error() {
        assert!(matches!(ET_Error::ET_Error_Ok.rs(), Ok(())));
        assert!(matches!(
            ET_Error::ET_Error_Internal.rs(),
            Err(Error::Internal)
        ));
        assert!(matches!(
            ET_Error::ET_Error_InvalidState.rs(),
            Err(Error::InvalidState)
        ));
        assert!(matches!(
            ET_Error::ET_Error_EndOfMethod.rs(),
            Err(Error::EndOfMethod)
        ));
        assert!(matches!(
            ET_Error::ET_Error_AlreadyLoaded.rs(),
            Err(Error::AlreadyLoaded)
        ));
        assert!(matches!(
            ET_Error::ET_Error_NotSupported.rs(),
            Err(Error::NotSupported)
        ));
        assert!(matches!(
            ET_Error::ET_Error_NotImplemented.rs(),
            Err(Error::NotImplemented)
        ));
        assert!(matches!(
            ET_Error::ET_Error_InvalidArgument.rs(),
            Err(Error::InvalidArgument)
        ));
        assert!(matches!(
            ET_Error::ET_Error_InvalidType.rs(),
            Err(Error::InvalidType)
        ));
        assert!(matches!(
            ET_Error::ET_Error_OperatorMissing.rs(),
            Err(Error::OperatorMissing)
        ));
        assert!(matches!(
            ET_Error::ET_Error_NotFound.rs(),
            Err(Error::NotFound)
        ));
        assert!(matches!(
            ET_Error::ET_Error_MemoryAllocationFailed.rs(),
            Err(Error::MemoryAllocationFailed)
        ));
        assert!(matches!(
            ET_Error::ET_Error_AccessFailed.rs(),
            Err(Error::AccessFailed)
        ));
        assert!(matches!(
            ET_Error::ET_Error_InvalidProgram.rs(),
            Err(Error::InvalidProgram)
        ));
        assert!(matches!(
            ET_Error::ET_Error_InvalidExternalData.rs(),
            Err(Error::InvalidExternalData)
        ));
        assert!(matches!(
            ET_Error::ET_Error_OutOfResources.rs(),
            Err(Error::OutOfResources)
        ));
        assert!(matches!(
            ET_Error::ET_Error_DelegateInvalidCompatibility.rs(),
            Err(Error::DelegateInvalidCompatibility)
        ));
        assert!(matches!(
            ET_Error::ET_Error_DelegateMemoryAllocationFailed.rs(),
            Err(Error::DelegateMemoryAllocationFailed)
        ));
        assert!(matches!(
            ET_Error::ET_Error_DelegateInvalidHandle.rs(),
            Err(Error::DelegateInvalidHandle)
        ));
    }
}
