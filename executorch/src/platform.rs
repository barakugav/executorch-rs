//! Platform abstraction layer to allow individual platform libraries to override
//! symbols in ExecuTorch.
//!
//! PAL functions are defined as C functions so a platform library implementer can use C in lieu of C++.

use crate::sys;
use core::ffi::CStr;

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
///
/// TODO: mark this function as unsafe
pub fn pal_init() {
    unsafe { sys::executorch_pal_init() };
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
            let msg =
                core::ffi::CStr::from_bytes_with_nul(&msg_buf.buf[..msg_buf.len + 1]).unwrap();
            (msg, msg_buf.len)
        }
        _ => (c"? (format error)", 1),
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
        super::pal_init();
    }
}
