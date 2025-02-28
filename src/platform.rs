//! Platform abstraction layer to allow individual platform libraries to override
//! symbols in ExecuTorch.
//!
//! PAL functions are defined as C functions so a platform library implementer can use C in lieu of C++.

/// Initialize the platform abstraction layer.
///
/// This function should be called before any other function provided by the PAL
/// to initialize any global state. Typically overridden by PAL implementer.
pub fn pal_init() {
    unsafe { executorch_sys::executorch_pal_init() };
}
