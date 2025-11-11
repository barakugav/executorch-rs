//! Platform abstraction layer to allow individual platform libraries to override
//! symbols in ExecuTorch.
//!
//! PAL functions are defined as C functions so a platform library implementer can use C in lieu of C++.

use core::ffi::CStr;
use core::ops::Not;
use core::ptr::NonNull;
use executorch_sys as sys;

/// Initialize the platform abstraction layer.
///
/// This function should be called before any other function provided by the PAL
/// to initialize any global state. Typically overridden by PAL implementer.
///
/// # Safety
///
/// This function should be called only once, before any accesses to the platform functions.
/// This function changes global state, but does not provide any synchronization.
/// Use this function carefully.
pub unsafe fn pal_init() {
    unsafe { sys::executorch_pal_init() };
}

/// Override the PAL functions with user implementations.
///
/// Any null entries in the table are unchanged and will keep the default implementation.
///
/// This function also calls the new platform's init function, if overridden.
///
/// # Returns
///
/// true if the registration was successful, false otherwise.
///
/// # Safety
///
/// This function changes global state, but does not provide any synchronization.
/// It should be called only once, before any accesses to the platform functions.
/// Use this function carefully.
pub unsafe fn register_platform_impl(plat_impl: PlatformImpl) -> bool {
    unsafe { sys::executorch_register_pal(plat_impl.0) }
}

/// Table of pointers to platform abstraction layer functions.
#[repr(transparent)]
pub struct PlatformImpl(sys::ExecutorchPalImpl);
impl PlatformImpl {
    /// Create a new, empty PlatformImpl.
    ///
    /// By default, all function pointers are null, meaning the default implementations will be used.
    ///
    /// # Arguments
    ///
    /// * `source_filename` - Optional C string representing the source filename the platform implementation is defined
    ///   in. In case of multiple platform implementations being registered, this can help in debugging.
    ///   To obtain a `&'static CStr` for the current file, you can use
    ///   `CStr::from_bytes_with_nul_unchecked(concat!(file!(), "\0").as_bytes())`.
    pub fn new(source_filename: Option<&'static CStr>) -> Self {
        Self(sys::ExecutorchPalImpl {
            init: None,
            abort: None,
            current_ticks: None,
            ticks_to_ns_multiplier: None,
            emit_log_message: None,
            allocate: None,
            free: None,
            source_filename: source_filename
                .map(|f| f.as_ptr())
                .unwrap_or(core::ptr::null()),
        })
    }

    /// Set the init function.
    ///
    /// The init function initialize the platform abstraction layer.
    /// It should be called before any other function provided by the PAL to initialize any global state.
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_init<F>(&mut self, f: F)
    where
        F: Fn() + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>()
        where
            F: Fn() + Copy + Send + Sync + 'static,
        {
            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f()
        }
        self.0.init = Some(f_impl::<F>);
        let _ = f;
    }

    /// Set the abort function.
    ///
    /// The abort function immediately abort execution, setting the device into an error state, if available.
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_abort<F>(&mut self, f: F)
    where
        F: Fn() + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>()
        where
            F: Fn() + Copy + Send + Sync + 'static,
        {
            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f()
        }
        self.0.abort = Some(f_impl::<F>);
        let _ = f;
    }

    /// Set the current_ticks function.
    ///
    /// The current_ticks function returns a monotonically non-decreasing timestamp in system ticks.
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_current_ticks<F>(&mut self, f: F)
    where
        F: Fn() -> Timestamp + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>() -> sys::executorch_timestamp_t
        where
            F: Fn() -> Timestamp + Copy + Send + Sync + 'static,
        {
            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f().0
        }
        self.0.current_ticks = Some(f_impl::<F>);
        let _ = f;
    }

    /// Set the ticks_to_ns_multiplier function.
    ///
    /// The ticks_to_ns_multiplier function returns the conversion rate from system ticks to nanoseconds as a fraction.
    /// To convert a system ticks to nanoseconds, multiply the tick count by the numerator and then divide by the
    /// denominator:
    /// ```python,ignore
    ///   nanoseconds = ticks * numerator / denominator
    /// ```
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_ticks_to_ns_multiplier<F>(&mut self, f: F)
    where
        F: Fn() -> TickRatio + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>() -> sys::executorch_tick_ratio
        where
            F: Fn() -> TickRatio + Copy + Send + Sync + 'static,
        {
            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f().0
        }
        self.0.ticks_to_ns_multiplier = Some(f_impl::<F>);
        let _ = f;
    }

    /// Set the emit_log_message function.
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_emit_log_message<F>(&mut self, f: F)
    where
        F: Fn(LogEntry) + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>(
            timestamp: sys::executorch_timestamp_t,
            level: sys::executorch_pal_log_level,
            filename: *const ::core::ffi::c_char,
            function: *const ::core::ffi::c_char,
            line: usize,
            message: *const ::core::ffi::c_char,
            length: usize,
        ) where
            F: Fn(LogEntry) + Copy + Send + Sync + 'static,
        {
            let timestamp = Timestamp(timestamp);
            let level = match level {
                sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_DEBUG => LogLevel::Debug,
                sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_INFO => LogLevel::Info,
                sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_ERROR => LogLevel::Error,
                sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_FATAL => LogLevel::Fatal,
                _ => LogLevel::Unknown,
            };
            let filename = filename
                .is_null()
                .not()
                .then(|| unsafe { CStr::from_ptr(filename).to_str().ok() })
                .flatten();
            let function = function
                .is_null()
                .not()
                .then(|| unsafe { CStr::from_ptr(function).to_str().ok() })
                .flatten();
            let message = message
                .is_null()
                .not()
                .then(|| {
                    let slice = unsafe { core::slice::from_raw_parts(message.cast(), length) };
                    core::str::from_utf8(slice).ok()
                })
                .flatten()
                .unwrap_or("? (invalid utf8 str)");

            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f(LogEntry {
                timestamp,
                level,
                filename,
                function,
                line,
                message,
            })
        }
        self.0.emit_log_message = Some(f_impl::<F>);
        let _ = f;
    }

    /// Set the allocate function.
    ///
    /// The allocate function allocates a block of memory of the given size in bytes and returns a pointer to it.
    /// May return `None` if the allocation fails.
    /// Memory allocated by this function will be freed later by a call to the platform's `free` function.
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_allocate<F>(&mut self, f: F)
    where
        F: Fn(usize) -> Option<NonNull<core::ffi::c_void>> + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>(size: usize) -> *mut core::ffi::c_void
        where
            F: Fn(usize) -> Option<NonNull<core::ffi::c_void>> + Copy + Send + Sync + 'static,
        {
            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f(size)
                .map(|ptr| ptr.as_ptr())
                .unwrap_or(core::ptr::null_mut())
        }
        self.0.allocate = Some(f_impl::<F>);
        let _ = f;
    }

    /// Set the free function.
    ///
    /// The free function frees a block of memory previously allocated by the platform's `allocate` function.
    ///
    /// The closure will be passed through FFI and called later by the Cpp ExecuTorch library.
    /// The closure must be zero-sized, trivially copyable (and dropped), Send, Sync, and 'static.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided closure is not zero-sized.
    pub fn set_free<F>(&mut self, f: F)
    where
        F: Fn(*mut core::ffi::c_void) + Copy + Send + Sync + 'static,
    {
        Self::check_closure_is_valid_for_ffi::<F>();
        unsafe extern "C" fn f_impl<F>(ptr: *mut core::ffi::c_void)
        where
            F: Fn(*mut core::ffi::c_void) + Copy + Send + Sync + 'static,
        {
            let f = PlatformImpl::closure_out_of_thin_air::<F>();
            f(ptr)
        }
        self.0.free = Some(f_impl::<F>);
        let _ = f;
    }

    fn check_closure_is_valid_for_ffi<F: Copy + Send + Sync + 'static>() {
        assert_eq!(
            core::mem::size_of::<F>(),
            0,
            "Closure must be zero-sized to be used in FFI"
        );
    }

    fn closure_out_of_thin_air<F: Copy + Send + Sync + 'static>() -> &'static F {
        debug_assert_eq!(core::mem::size_of::<F>(), 0);
        // Safety: the closure is zero-sized, copy (and no drop), send, sync and 'static
        unsafe { NonNull::<F>::dangling().as_ref() }
    }
}
impl Clone for PlatformImpl {
    fn clone(&self) -> Self {
        Self(sys::ExecutorchPalImpl {
            init: self.0.init,
            abort: self.0.abort,
            current_ticks: self.0.current_ticks,
            ticks_to_ns_multiplier: self.0.ticks_to_ns_multiplier,
            emit_log_message: self.0.emit_log_message,
            allocate: self.0.allocate,
            free: self.0.free,
            source_filename: self.0.source_filename,
        })
    }
}

/// Platform timestamp in system ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Timestamp(sys::executorch_timestamp_t);
impl Timestamp {
    /// Create a new Timestamp from the given tick count.
    pub fn new(ticks: u64) -> Self {
        Self(ticks)
    }

    /// Get the tick count of this Timestamp.
    pub fn ticks(&self) -> u64 {
        self.0
    }
}

/// Represents the conversion ratio from system ticks to nanoseconds.
///
/// To convert, use nanoseconds = ticks * numerator / denominator.
#[derive(Debug)]
#[repr(transparent)]
pub struct TickRatio(sys::executorch_tick_ratio);
impl TickRatio {
    /// Create a new TickRatio with the given numerator and denominator.
    pub fn new(numerator: u64, denominator: u64) -> Self {
        Self(sys::executorch_tick_ratio {
            numerator,
            denominator,
        })
    }

    /// Get the numerator of this TickRatio.
    pub fn numerator(&self) -> u64 {
        self.0.numerator
    }

    /// Get the denominator of this TickRatio.
    pub fn denominator(&self) -> u64 {
        self.0.denominator
    }
}

/// Severity level of a log message
#[repr(u32)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum LogLevel {
    Debug = sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_DEBUG as u32,
    Info = sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_INFO as u32,
    Error = sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_ERROR as u32,
    Fatal = sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_FATAL as u32,
    Unknown = sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_UNKNOWN as u32,
}

/// A log entry, consumed by the platform log message handler.
#[derive(Debug)]
pub struct LogEntry<'a> {
    /// Timestamp of the log message.
    pub timestamp: Timestamp,
    /// Log level of the message.
    pub level: LogLevel,
    /// Filename where the log message was emitted, if available.
    pub filename: Option<&'a str>,
    /// Function name where the log message was emitted, if available.
    pub function: Option<&'a str>,
    /// Line number where the log message was emitted.
    pub line: usize,
    /// The log message.
    pub message: &'a str,
}

pub(crate) fn emit_log(
    timestamp: sys::executorch_timestamp_t,
    level: sys::executorch_pal_log_level,
    filename: &CStr,
    function: &CStr,
    line: usize,
    msg_args: core::fmt::Arguments,
) {
    const MAX_LOG_MESSAGE_LEN: usize = 256;
    pub struct FormatBuffer {
        buf: [u8; MAX_LOG_MESSAGE_LEN],
        len: usize,
    }
    impl core::fmt::Write for FormatBuffer {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let s = s.as_bytes();
            let remaining = &mut self.buf[self.len..];
            if s.len() > remaining.len() {
                return Err(core::fmt::Error); // not enough space
            }
            remaining[..s.len()].copy_from_slice(s);
            self.len += s.len();
            Ok(())
        }
    }

    let mut msg_buf = FormatBuffer {
        buf: [0; MAX_LOG_MESSAGE_LEN],
        len: 0,
    };
    let fmt_res = core::fmt::write(&mut msg_buf, msg_args);
    let (msg, msg_len) = match fmt_res {
        Ok(()) if msg_buf.len < MAX_LOG_MESSAGE_LEN - 1 => {
            msg_buf.buf[msg_buf.len] = 0;
            // Safety: we just wrote null byte at the end
            let msg = unsafe {
                core::ffi::CStr::from_bytes_with_nul_unchecked(&msg_buf.buf[..msg_buf.len + 1])
            };
            (msg, msg_buf.len)
        }
        _ => {
            let msg_bytes = b"? (format error)\0";
            // Safety: there is a `\0` at the end of the message bytes
            let msg = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(msg_bytes) };
            (msg, msg_bytes.len() - 1)
        }
    };

    unsafe {
        sys::executorch_pal_emit_log_message(
            timestamp,
            level,
            filename.as_ptr(),
            function.as_ptr(),
            line,
            msg.as_ptr(),
            msg_len,
        )
    }
}

#[cfg(test)]
mod tests {

    #[ctor::ctor]
    fn pal_init() {
        // Safety: we call pal_init once, before any other executorch operations, and before any thread is spawned
        unsafe { super::pal_init() };
    }
}
