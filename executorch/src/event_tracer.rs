//! Event tracers for debugging and profiling.

use std::marker::PhantomData;

/// EventTracer is a class that users can inherit and implement to log/serialize/stream etc.
///
/// The profiling and debugging events that are generated at runtime for a model. An example of this is the ETDump
/// implementation in the devtools codebase that serializes these events to a flatbuffer.
pub struct EventTracer<'a>([std::ffi::c_void; 0], PhantomData<&'a ()>);

/// A unique pointer to an EventTracer.
#[cfg(feature = "module")]
pub struct EventTracerPtr<'a>(
    pub(crate) executorch_sys::cxx::UniquePtr<executorch_sys::EventTracer>,
    PhantomData<&'a ()>,
);

#[cfg(feature = "etdump")]
pub use etdump::*;
#[cfg(feature = "etdump")]
mod etdump {
    use std::marker::PhantomData;

    use executorch_sys as sys;

    use super::EventTracer;

    /// ETDump (ExecuTorch Dump) is one of the core components of the ExecuTorch Developer Tools.
    ///
    /// It is the mechanism through which all forms of profiling and debugging data is extracted from the runtime.
    /// Users can’t parse ETDump directly; instead, they should pass it into the Inspector API, which deserializes the data,
    /// offering interfaces for flexible analysis and debugging.
    pub struct ETDumpGen<'a>(sys::ETDumpGen, PhantomData<&'a ()>);
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
            let buffer = sys::SpanU8 { data, len };

            let self_ = unsafe { sys::executorch_ETDumpGen_new(buffer) };
            Self(self_, PhantomData)
        }

        /// Create a new ETDumpGen object using the given buffer.
        pub fn new_in_buffer(buffer: &'a mut [u8]) -> Self {
            Self::new_impl(Some(buffer))
        }

        /// Get the ETDump data.
        pub fn get_etdump_data(&mut self) -> Option<&[u8]> {
            let data =
                unsafe { sys::executorch_ETDumpGen_get_etdump_data((&mut self.0) as *mut _) };
            if data.data.is_null() {
                None
            } else {
                Some(unsafe { std::slice::from_raw_parts(data.data, data.len) })
            }
        }

        fn as_event_tracer_ptr(&self) -> *const EventTracer<'a> {
            let self_ = (&self.0) as *const _ as *mut sys::ETDumpGen;
            let tracer = unsafe { sys::executorch_ETDumpGen_as_event_tracer_mut(self_) };
            let tracer = tracer.ptr as *mut EventTracer<'a>;
            tracer as *const _
        }
    }
    impl<'a> AsRef<EventTracer<'a>> for ETDumpGen<'a> {
        fn as_ref(&self) -> &EventTracer<'a> {
            unsafe { &*self.as_event_tracer_ptr() }
        }
    }
    impl<'a> AsMut<EventTracer<'a>> for ETDumpGen<'a> {
        fn as_mut(&mut self) -> &mut EventTracer<'a> {
            unsafe { &mut *self.as_event_tracer_ptr().cast_mut() }
        }
    }
}
