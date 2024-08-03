//! Utility functions and types.
//!
//! Most of the structs in this module may seems redundant in Rust, but they are wrappers around C++ types
//! that are used in the C++ API. Some structs and functions accept these types as arguments, so they are
//! necessary to interact with the C++ API.

use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use crate::{et_c, et_rs_c};

pub(crate) trait IntoRust {
    type RsType;
    fn rs(self) -> Self::RsType;
}

/// Represents a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// Span and ArrayRef are extremely similar with the difference being ArrayRef
/// views a list of constant elements and Span views a list of mutable elements.
/// Clients should decide between the two based on if the list elements for their
/// use case should be mutable.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
#[allow(dead_code)]
pub struct ArrayRef<'a, T>(pub(crate) et_c::ArrayRef<T>, PhantomData<&'a ()>);
impl<'a, T> ArrayRef<'a, T> {
    pub(crate) unsafe fn from_inner(arr: &et_c::ArrayRef<T>) -> Self {
        Self(
            et_c::ArrayRef::<T> {
                Data: arr.Data,
                Length: arr.Length,
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }

    /// Create an ArrayRef from a slice.
    ///
    /// The given slice must outlive the ArrayRef.
    pub fn from_slice(s: &'a [T]) -> Self {
        Self(
            et_c::ArrayRef {
                Data: s.as_ptr(),
                Length: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.0.Data, self.0.Length) }
    }
}
impl<T: Debug> Debug for ArrayRef<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// Represent a reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the Span.
///
/// Span and ArrayRef are extremely similar with the difference being ArrayRef
/// views a list of constant elements and Span views a list of mutable elements.
/// Clients should decide between the two based on if the list elements for their
/// use case should be mutable.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
#[allow(dead_code)]
pub struct Span<'a, T>(pub(crate) et_c::Span<T>, PhantomData<&'a T>);
impl<'a, T> Span<'a, T> {
    pub(crate) unsafe fn new(span: et_c::Span<T>) -> Self {
        Self(span, PhantomData)
    }

    /// Create a Span from a mutable slice.
    ///
    /// The given slice must outlive the Span.
    pub fn from_slice(s: &'a mut [T]) -> Self {
        Self(
            et_c::Span {
                data_: s.as_mut_ptr(),
                length_: s.len(),
                _phantom_0: PhantomData,
            },
            PhantomData,
        )
    }

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.data_, self.0.length_) }
    }
}
impl<T: Debug> Debug for Span<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// Leaner optional class, subset of c10, std, and boost optional APIs.
pub struct Optional<T>(et_c::optional<T>);
impl<T> Optional<T> {
    /// Create a new Optional based on the given Option.
    pub fn new(val: Option<T>) -> Self {
        let is_some = val.is_some();
        Self(et_c::optional::<T> {
            trivial_init: et_c::optional_trivial_init_t { _address: 0 },
            storage_: val
                .map(|value| et_c::optional_storage_t {
                    value_: ManuallyDrop::new(value),
                })
                .unwrap_or(et_c::optional_storage_t {
                    dummy_: ManuallyDrop::new(0),
                }),
            init_: is_some,
            _phantom_0: PhantomData,
        })
    }

    /// Get an optional reference to the value.
    pub fn as_ref(&self) -> Option<&T> {
        self.0.init_.then(|| unsafe { &*self.0.storage_.value_ })
    }

    /// Convert this Optional into an Option.
    pub fn into_option(mut self) -> Option<T> {
        self.0.init_.then(|| {
            self.0.init_ = false;
            unsafe { ManuallyDrop::take(&mut self.0.storage_.value_) }
        })
    }
}
impl<T> Drop for Optional<T> {
    fn drop(&mut self) {
        if self.0.init_ {
            unsafe {
                ManuallyDrop::drop(&mut self.0.storage_.value_);
            }
        }
    }
}
impl<T> From<Optional<T>> for Option<T> {
    fn from(opt: Optional<T>) -> Option<T> {
        opt.into_option()
    }
}
impl<T> From<Option<T>> for Optional<T> {
    fn from(opt: Option<T>) -> Optional<T> {
        Optional::new(opt)
    }
}
impl<T: Clone> Clone for Optional<T> {
    fn clone(&self) -> Self {
        Self::new(self.as_ref().cloned())
    }
}
impl<T: PartialEq> PartialEq for Optional<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl<T: Eq> Eq for Optional<T> {}
impl<T: Hash> Hash for Optional<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}
impl<T: Debug> Debug for Optional<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

#[allow(dead_code)]
pub(crate) fn str2chars(s: &str) -> Result<&[std::os::raw::c_char], &'static str> {
    let bytes = s.as_bytes();
    if bytes.iter().any(|&b| b == 0) {
        return Err("String contains null byte");
    }
    let chars = bytes.as_ptr().cast::<std::os::raw::c_char>();
    Ok(unsafe { std::slice::from_raw_parts(chars, bytes.len()) })
}
#[allow(dead_code)]
pub(crate) fn chars2string(chars: Vec<std::os::raw::c_char>) -> String {
    let bytes = unsafe { std::mem::transmute::<Vec<std::os::raw::c_char>, Vec<u8>>(chars) };
    String::from_utf8(bytes).unwrap()
}

impl<T> IntoRust for et_rs_c::RawVec<T> {
    type RsType = Vec<T>;
    fn rs(self) -> Self::RsType {
        unsafe { Vec::from_raw_parts(self.data, self.len, self.cap) }
    }
}

// Debug func
#[allow(dead_code)]
pub(crate) fn to_bytes<T>(val: &T) -> Vec<u8> {
    (0..std::mem::size_of_val(val))
        .map(|i| unsafe {
            let ptr = val as *const T as *const u8;
            *ptr.add(i)
        })
        .collect()
}

/// Transmute from A to B.
///
/// Like transmute, but does not have the compile-time size check which blocks
/// using regular transmute in some cases.
///
/// **Panics** if the size of A and B are different.
#[inline]
pub(crate) unsafe fn unlimited_transmute<A, B>(data: A) -> B {
    // safe when sizes are equal and caller guarantees that representations are equal
    assert_eq!(std::mem::size_of::<A>(), std::mem::size_of::<B>());
    let old_data = ManuallyDrop::new(data);
    (&*old_data as *const A as *const B).read()
}
