//! Event tracers for debugging and profiling.

use std::marker::PhantomData;

#[cfg(feature = "module")]
use executorch_sys as et_c;

/// EventTracer is a class that users can inherit and implement to log/serialize/stream etc.
///
/// The profiling and debugging events that are generated at runtime for a model. An example of this is the ETDump
/// implementation in the devtools codebase that serializes these events to a flatbuffer.
pub struct EventTracer<'a>([std::ffi::c_void; 0], PhantomData<&'a ()>);

/// A unique pointer to an EventTracer.
#[cfg(feature = "module")]
pub struct EventTracerPtr<'a>(
    pub(crate) et_c::cxx::UniquePtr<et_c::cpp::EventTracer>,
    PhantomData<&'a ()>,
);

#[cfg(feature = "etdump")]
pub use etdump::*;
#[cfg(feature = "etdump")]
mod etdump {
    use std::marker::PhantomData;

    use executorch_sys as et_c;

    use super::EventTracer;

    /// ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch Developer Tools.
    ///
    /// It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime.
    /// Users can’t parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data,
    /// offering interfaces for flexible analysis and debugging.
    pub struct ETDumpGen<'a>(et_c::ETDumpGen, PhantomData<&'a ()>);
    #[cfg(feature = "std")]
    impl Default for ETDumpGen<'static> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ETDumpGen<'static> {
        /// Create a new ETDumpGen object with a buffer allocated using malloc.
        #[cfg(feature = "std")]
        pub fn new() -> Self {
            Self::new_impl(None)
        }
    }
    impl<'a> ETDumpGen<'a> {
        fn new_impl(buffer: Option<&'a mut [u8]>) -> Self {
            let (data, len) = buffer
                .map(|b| (b.as_mut_ptr(), b.len()))
                .unwrap_or((std::ptr::null_mut(), 0));
            let buffer = et_c::SpanU8 { data, len };

            let self_ = unsafe { et_c::executorch_ETDumpGen_new(buffer) };
            Self(self_, PhantomData)
        }

        /// Create a new ETDumpGen object using the given buffer.
        pub fn new_in_buffer(buffer: &'a mut [u8]) -> Self {
            Self::new_impl(Some(buffer))
        }

        /// Get the ETDump data.
        pub fn get_etdump_data(&mut self) -> Option<&[u8]> {
            let data =
                unsafe { et_c::executorch_ETDumpGen_get_etdump_data((&mut self.0) as *mut _) };
            if data.data.is_null() {
                None
            } else {
                Some(unsafe { std::slice::from_raw_parts(data.data, data.len) })
            }
        }
    }
    impl<'a> AsMut<EventTracer<'a>> for ETDumpGen<'a> {
        fn as_mut(&mut self) -> &mut EventTracer<'a> {
            let self_ = (&mut self.0) as *mut et_c::ETDumpGen;
            let tracer = unsafe { et_c::executorch_ETDumpGen_as_event_tracer_mut(self_) };
            let tracer = tracer.ptr as *mut EventTracer<'a>;
            unsafe { &mut *tracer }
        }
    }
}
