macro_rules! file_cstr {
    () => {{
        const BYTES: &[u8] = concat!(file!(), "\0").as_bytes();
        // SAFETY: We just added a null terminator and file!() doesn't contain nulls
        unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(BYTES) }
    }};
}

macro_rules! log {
    ($level:ident, $($arg:tt)*) => {{
        let timestamp = unsafe { executorch_sys::executorch_pal_current_ticks() };
        let level = executorch_sys::executorch_pal_log_level::$level;
        let filename = crate::log::file_cstr!();
        let function = c"";
        let line = line!() as usize;
        crate::platform::emit_log(timestamp, level, filename, function, line, core::format_args!($($arg)*));
    }}
}
macro_rules! error {
    ($($arg:tt)*) => {{
        crate::log::log!(EXECUTORCH_PAL_LOG_LEVEL_ERROR, $($arg)*);
    }}
}
pub(crate) use error;
pub(crate) use file_cstr;
pub(crate) use log;
