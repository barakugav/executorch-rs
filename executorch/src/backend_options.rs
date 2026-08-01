//! Per-backend load-time configuration options.
//!
//! Build a set of [`BackendOption`]s, associate them with a backend id in a
//! [`LoadBackendOptionsMap`], and pass the map to
//! [`Program::load_method`](crate::program::Program::load_method) or the `Module` load
//! methods to configure per-delegate behavior at load time.

use core::ffi::{c_char, c_int, CStr};
use core::marker::PhantomData;
use std::mem::MaybeUninit;

use executorch_sys as sys;

use crate::util::{c_new, str2chars, try_c_new, ArrayRef, IntoRust};
use crate::{Error, Result};

/// A single per-backend key/value option, such as `num_threads = 4`.
///
/// Keys are limited to 63 bytes and string values to 255 bytes; the constructors return
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) if exceeded.
#[repr(transparent)]
pub struct BackendOption(sys::ET_BackendOption);
impl BackendOption {
    /// Create a boolean option. Fails if `key` is longer than 63 bytes.
    pub fn new_bool(key: &str, value: bool) -> Result<Self> {
        let key = ArrayRef::from_chars(str2chars(key));
        // Safety: executorch_BackendOption_new_bool writes to the out pointer on success.
        unsafe { try_c_new(|out| sys::executorch_BackendOption_new_bool(key.0, value, out)) }
            .map(Self)
    }

    /// Create an integer option.
    ///
    /// Fails if `key` is longer than 63 bytes, or if `value` does not fit in a C `int`.
    pub fn new_int(key: &str, value: i64) -> Result<Self> {
        let value: c_int = value.try_into().map_err(|_| Error::InvalidArgument)?;
        let key = ArrayRef::from_chars(str2chars(key));
        // Safety: executorch_BackendOption_new_int writes to the out pointer on success.
        unsafe { try_c_new(|out| sys::executorch_BackendOption_new_int(key.0, value, out)) }
            .map(Self)
    }

    /// Create a string option. Fails if `key` is longer than 63 bytes or `value` longer than 255.
    pub fn new_str(key: &str, value: &str) -> Result<Self> {
        let key = ArrayRef::from_chars(str2chars(key));
        let value = ArrayRef::from_chars(str2chars(value));
        // Safety: executorch_BackendOption_new_str writes to the out pointer on success.
        unsafe { try_c_new(|out| sys::executorch_BackendOption_new_str(key.0, value.0, out)) }
            .map(Self)
    }

    /// The option key.
    pub fn key(&self) -> &str {
        // Safety: the returned pointer is a nul-terminated string within `self`'s key buffer.
        let key = unsafe { CStr::from_ptr(sys::executorch_BackendOption_key(&self.0)) };
        key.to_str().unwrap_or("")
    }

    /// Returns `true` if this option holds a bool value.
    pub fn is_bool(&self) -> bool {
        unsafe { sys::executorch_BackendOption_is_bool(&self.0) }
    }

    /// Returns `true` if this option holds an int value.
    pub fn is_int(&self) -> bool {
        unsafe { sys::executorch_BackendOption_is_int(&self.0) }
    }

    /// Returns `true` if this option holds a string value.
    pub fn is_str(&self) -> bool {
        unsafe { sys::executorch_BackendOption_is_str(&self.0) }
    }

    /// The value as a bool, or `None` if it is not a bool option.
    pub fn as_bool(&self) -> Option<bool> {
        // Safety: executorch_BackendOption_as_bool writes to the pointer on success.
        unsafe { try_c_new(|out| sys::executorch_BackendOption_as_bool(&self.0, out)).ok() }
    }

    /// The value as an int, or `None` if it is not an int option.
    pub fn as_int(&self) -> Option<i64> {
        // Safety: executorch_BackendOption_as_int writes to the pointer on success.
        unsafe { try_c_new(|out| sys::executorch_BackendOption_as_int(&self.0, out)).ok() }
            .map(|v| v as i64)
    }

    /// The value as a string, or `None` if it is not a string option.
    pub fn as_str(&self) -> Option<&str> {
        // Safety: executorch_BackendOption_as_str writes to the pointer on success.
        let ptr =
            unsafe { try_c_new(|out| sys::executorch_BackendOption_as_str(&self.0, out)).ok()? };
        Some(unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or(""))
    }
}

/// Maps backend IDs to their load-time options.
///
/// This class is used to provide per-delegate configuration at Module::load()
/// time. Users can set options for multiple backends, and the runtime will
/// route the appropriate options to each backend during initialization.
///
/// Note: This class does NOT take ownership of the option spans. The caller
/// must ensure that the BackendOptions objects outlive the LoadBackendOptionsMap
/// and any loaded models that use it.
#[repr(transparent)]
pub struct LoadBackendOptionsMap<'a>(sys::ET_LoadBackendOptionsMap, PhantomData<&'a ()>);
impl<'a> LoadBackendOptionsMap<'a> {
    /// Create an empty map.
    pub fn new() -> Self {
        // Safety: executorch_LoadBackendOptionsMap_new writes to the out pointer on success.
        let inner = unsafe { c_new(|out| sys::executorch_LoadBackendOptionsMap_new(out)) };
        Self(inner, PhantomData)
    }

    /// Sets options for a specific backend.
    ///
    /// If options for the given backend_id already exist, they will be replaced.
    ///
    /// # Arguments
    ///
    /// - `backend_id`: The backend identifier (e.g., "CoreMLBackend", "XNNPACKBackend"). Must not
    ///   be empty.
    /// - `options`: Span of BackendOption to associate with this backend. The span's underlying
    ///   data must outlive this map and any models loaded with it.
    pub fn set_options(&mut self, backend_id: &str, options: &'a [BackendOption]) -> Result<()> {
        let backend_id = ArrayRef::from_chars(str2chars(backend_id));
        // Safety: BackendOption is #[repr(transparent)]
        let ptr = options.as_ptr().cast::<sys::ET_BackendOption>();
        unsafe {
            sys::executorch_LoadBackendOptionsMap_set_options(
                &mut self.0,
                backend_id.0,
                ptr,
                options.len(),
            )
        }
        .rs()
    }

    /// Returns the number of backends with configured options.
    pub fn len(&self) -> usize {
        unsafe { sys::executorch_LoadBackendOptionsMap_size(&self.0) }
    }

    /// Whether the map has no configured backends.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the (backend_id, options) entry at the given index for enumeration over the map's contents.
    pub fn get(&self, index: usize) -> Result<(&str, &[BackendOption])> {
        let (backend_id, options, n_options) = unsafe {
            let mut backend_id = MaybeUninit::<*const c_char>::uninit();
            let mut options = MaybeUninit::<*const sys::ET_BackendOption>::uninit();
            let mut n_options = MaybeUninit::<usize>::uninit();
            sys::executorch_LoadBackendOptionsMap_entry_at(
                &self.0,
                index,
                backend_id.as_mut_ptr(),
                options.as_mut_ptr(),
                n_options.as_mut_ptr(),
            )
            .rs()?;
            (
                backend_id.assume_init(),
                options.assume_init(),
                n_options.assume_init(),
            )
        };

        let id = unsafe { CStr::from_ptr(backend_id) }
            .to_str()
            .map_err(|_| Error::InvalidString)?;
        // Safety: BackendOption is #[repr(transparent)]
        let options = options.cast::<BackendOption>();
        let options = unsafe { core::slice::from_raw_parts(options, n_options) };

        Ok((id, options))
    }

    pub(crate) fn as_cpp_ptr(&self) -> *const sys::ET_LoadBackendOptionsMap {
        &self.0
    }
}
impl Default for LoadBackendOptionsMap<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_option_bool() {
        let opt = BackendOption::new_bool("enable_profiling", true).unwrap();
        assert_eq!(opt.key(), "enable_profiling");
        assert!(opt.is_bool());
        assert_eq!(opt.as_bool(), Some(true));
        assert_eq!(opt.as_int(), None);
        assert_eq!(opt.as_str(), None);
    }

    #[test]
    fn backend_option_int() {
        let opt = BackendOption::new_int("num_threads", 4).unwrap();
        assert_eq!(opt.key(), "num_threads");
        assert!(opt.is_int());
        assert_eq!(opt.as_int(), Some(4));
        assert_eq!(opt.as_bool(), None);
    }

    #[test]
    fn backend_option_str() {
        let opt = BackendOption::new_str("compute_unit", "cpu_and_gpu").unwrap();
        assert!(opt.is_str());
        assert_eq!(opt.as_str(), Some("cpu_and_gpu"));
        assert_eq!(opt.as_int(), None);
    }

    #[test]
    fn backend_option_key_too_long() {
        let long_key = core::str::from_utf8(&[b'k'; 64]).unwrap();
        assert!(BackendOption::new_bool(long_key, true).is_err());
    }

    #[test]
    fn backend_option_int_out_of_range() {
        assert!(BackendOption::new_int("x", i64::from(i32::MAX) + 1).is_err());
    }

    #[test]
    fn options_map_set_get() {
        let opts = [
            BackendOption::new_int("num_threads", 4).unwrap(),
            BackendOption::new_bool("enable_profiling", true).unwrap(),
        ];
        let mut map = LoadBackendOptionsMap::new();
        assert!(map.is_empty());
        map.set_options("XnnpackBackend", &opts).unwrap();
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());

        let (id, got) = map.get(0).unwrap();
        assert_eq!(id, "XnnpackBackend");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].key(), "num_threads");
        assert_eq!(got[0].as_int(), Some(4));
        assert_eq!(got[1].as_bool(), Some(true));
        assert!(map.get(1).is_err());
    }
}
