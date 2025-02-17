//! Error types used in the [`executortorch`](crate) crate.

use core::mem::MaybeUninit;

use crate::{et_c, util::IntoRust};

use et_c::runtime::Error as CError;

/// ExecuTorch Error type.
pub struct Error {
    inner: ErrorInner,
}
enum ErrorInner {
    Simple(ErrorKind),
}
impl Error {
    pub(crate) fn simple(kind: ErrorKind) -> Self {
        Self {
            inner: ErrorInner::Simple(kind),
        }
    }

    /// Get the kind of error.
    pub fn kind(&self) -> ErrorKind {
        match self.inner {
            ErrorInner::Simple(err) => err,
        }
    }
}
impl std::fmt::Debug for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.inner {
            ErrorInner::Simple(kind) => fmt.debug_tuple("Kind").field(&kind).finish(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.inner {
            ErrorInner::Simple(err) => std::fmt::Display::fmt(&err, fmt),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Categories of errors that can occur in executorch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum ErrorKind {
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
impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for ErrorKind {}

impl IntoRust for CError {
    type RsType = Result<()>;
    fn rs(self) -> Self::RsType {
        let kind = match self {
            CError::Ok => return Ok(()),
            CError::Internal => ErrorKind::Internal,
            CError::InvalidState => ErrorKind::InvalidState,
            CError::EndOfMethod => ErrorKind::EndOfMethod,
            CError::NotSupported => ErrorKind::NotSupported,
            CError::NotImplemented => ErrorKind::NotImplemented,
            CError::InvalidArgument => ErrorKind::InvalidArgument,
            CError::InvalidType => ErrorKind::InvalidType,
            CError::OperatorMissing => ErrorKind::OperatorMissing,
            CError::NotFound => ErrorKind::NotFound,
            CError::MemoryAllocationFailed => ErrorKind::MemoryAllocationFailed,
            CError::AccessFailed => ErrorKind::AccessFailed,
            CError::InvalidProgram => ErrorKind::InvalidProgram,
            CError::DelegateInvalidCompatibility => ErrorKind::DelegateInvalidCompatibility,
            CError::DelegateMemoryAllocationFailed => ErrorKind::DelegateMemoryAllocationFailed,
            CError::DelegateInvalidHandle => ErrorKind::DelegateInvalidHandle,
        };
        Err(Error::simple(kind))
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn try_new<T>(f: impl FnOnce(*mut T) -> CError) -> crate::Result<T> {
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
