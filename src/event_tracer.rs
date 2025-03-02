//! ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch Developer Tools.
//!
//! It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime.
//! Users can’t parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data,
//! offering interfaces for flexible analysis and debugging.
//!
//! Tracing is not fully supported yet.

use std::marker::PhantomData;

#[cfg(feature = "std")]
use executorch_sys as et_c;

/// EventTracer is a class that users can inherit and implement to log/serialize/stream etc.
///
/// The profiling and debugging events that are generated at runtime for a model. An example of this is the ETDump
/// implementation in the devtools codebase that serializes these events to a flatbuffer.
pub struct EventTracer<'a>([std::ffi::c_void; 0], PhantomData<&'a ()>);

/// A unique pointer to an EventTracer.
#[cfg(feature = "std")]
pub struct EventTracerPtr<'a>(
    pub(crate) et_c::cxx::UniquePtr<et_c::cpp::EventTracer>,
    PhantomData<&'a ()>,
);
